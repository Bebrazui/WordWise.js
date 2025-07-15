
// src/lib/model.ts
import { Embedding, Linear, LSTMCell, Layer, crossEntropyLossWithSoftmaxGrad, PositionalEmbedding } from './layers';
import { TransformerEncoderBlock } from './transformer';
import { FlowNetBlock } from './flownet';
import { SGD } from './optimizer';
import { Tensor } from './tensor';
import { createSequenceBatches } from '@/utils/batching';


type TextVocabDataType = {
  vocab: string[];
  wordToIndex: Map<string, number>;
  indexToWord: Map<number, string>;
  vocabSize: number;
};

type ImageVocabDataType = {
  labels: string[];
  labelToIndex: Map<string, number>;
  indexToLabel: Map<number, string>;
  numClasses: number;
}

export type VocabData = TextVocabDataType | ImageVocabDataType;


export interface FitCallbacks {
    onEpochEnd?: (log: { epoch: number; loss: number; gradients: { layer: string; avgGrad: number }[] }) => boolean | void;
}

// --- Base Model Class ---
abstract class BaseModelClass {
    abstract type: 'text' | 'image' | 'transformer' | 'flownet';
    public stopTraining: boolean = false;

    abstract getParameters(): Tensor[];
    abstract getLayers(): { [key: string]: Layer | Layer[] };
    abstract getArchitecture(): any;

    /**
     * Compiles and trains the model.
     * @param inputs Array of input Tensors.
     * @param targets Array of target Tensors.
     * @param options Training options like epochs, batchSize, learningRate.
     * @param callbacks Callbacks for logging progress.
     */
    async fit(
        inputs: Tensor[],
        targets: Tensor[],
        options: { epochs: number; batchSize: number; learningRate: number; initialEpoch?: number },
        callbacks?: FitCallbacks
    ): Promise<void> {

        const optimizer = new SGD(options.learningRate);
        const startEpoch = options.initialEpoch || 0;

        const seqLen = (this as any).seqLen || 1;
        const batches = createSequenceBatches(inputs, targets, options.batchSize, seqLen, this.type === 'transformer');

        if (batches.length === 0) {
            console.warn("Could not create batches. Check your data and sequence length.");
            return;
        }

        for (let epoch = 0; epoch < options.epochs; epoch++) {
            let epochLoss = 0;

            for (const batch of batches) {
                const batchInputs = new Tensor(batch.inputs.data, batch.inputs.shape);
                const batchTargets = new Tensor(batch.targets.data, batch.targets.shape);

                const predictionLogits = (this as any).forward(batchInputs).outputLogits;
                
                const loss = crossEntropyLossWithSoftmaxGrad(predictionLogits, batchTargets);
                if(loss.data.length === 1) {
                    epochLoss += loss.data[0];
                }

                loss.backward();
                optimizer.step(this.getParameters());
            }

            const avgEpochLoss = epochLoss / batches.length;
            const currentEpoch = startEpoch + epoch;

            if (callbacks?.onEpochEnd) {
                const gradientInfo = Object.entries(this.getLayers()).flatMap(([layerName, layerOrLayers]) => {
                    const layers = Array.isArray(layerOrLayers) ? layerOrLayers : [layerOrLayers];
                    return layers.map((layer, index) => {
                         let totalGradMag = 0;
                         let paramCount = 0;
                         layer.getParameters().forEach(p => {
                            if (p.grad) {
                               totalGradMag += p.grad.data.reduce((acc, val) => acc + Math.abs(val), 0);
                               paramCount += p.grad.size;
                            }
                         });
                         const avgGrad = paramCount > 0 ? totalGradMag / paramCount : 0;
                         const finalLayerName = layers.length > 1 ? `${layerName}_${index}` : layerName;
                         return { layer: finalLayerName, avgGrad };
                    });
                });

                const stop = callbacks.onEpochEnd({
                    epoch: currentEpoch,
                    loss: avgEpochLoss,
                    gradients: gradientInfo
                });
                if (stop) break;
            }
             // Give the main thread a chance to breathe
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }
}


/**
 * WordWiseModel: A simple recurrent neural network for text processing.
 */
export class WordWiseModel extends BaseModelClass {
  type: 'text' = 'text';
  embeddingLayer: Embedding;
  lstmCell: LSTMCell;
  outputLayer: Linear;
  public hiddenSize: number;
  public vocabSize: number;
  public embeddingDim: number;

  constructor(vocabSize: number, embeddingDim: number, hiddenSize: number) {
    super();
    this.embeddingLayer = new Embedding(vocabSize, embeddingDim);
    this.lstmCell = new LSTMCell(embeddingDim, hiddenSize);
    this.outputLayer = new Linear(hiddenSize, vocabSize);
    this.hiddenSize = hiddenSize;
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
  }

  forward(input: Tensor, prevH?: Tensor, prevC?: Tensor): { outputLogits: Tensor, h: Tensor, c: Tensor } {
    const batchSize = input.shape[0];
    const h = prevH || Tensor.zeros([batchSize, this.hiddenSize]);
    const c = prevC || Tensor.zeros([batchSize, this.hiddenSize]);
    
    const embeddedInput = this.embeddingLayer.forward(input);
    const { h: nextH, c: nextC } = this.lstmCell.forward(embeddedInput.reshape([-1, this.embeddingDim]), h, c);
    const outputLogits = this.outputLayer.forward(nextH);
    return { outputLogits, h: nextH, c: nextC };
  }

  getParameters(): Tensor[] {
    return [
      ...this.embeddingLayer.getParameters(),
      ...this.lstmCell.getParameters(),
      ...this.outputLayer.getParameters(),
    ];
  }
  
  getLayers(): { [key: string]: Layer } {
    return {
      'Embedding': this.embeddingLayer,
      'LSTM': this.lstmCell,
      'Output': this.outputLayer,
    }
  }

  getArchitecture() {
      return { type: 'lstm', vocabSize: this.vocabSize, embeddingDim: this.embeddingDim, hiddenSize: this.hiddenSize };
  }

  initializeStates(batchSize: number): { h0: Tensor, c0: Tensor } {
    const h0 = Tensor.zeros([batchSize, this.hiddenSize]);
    const c0 = Tensor.zeros([batchSize, this.hiddenSize]);
    return { h0, c0 };
  }
}

export class TransformerModel extends BaseModelClass {
    type: 'transformer' = 'transformer';
    embeddingLayer: Embedding;
    posEmbeddingLayer: PositionalEmbedding;
    transformerBlocks: TransformerEncoderBlock[];
    outputLayer: Linear;

    public vocabSize: number;
    public dModel: number;
    public numLayers: number;
    public seqLen: number;
    public dff: number;
    public numHeads: number;

    constructor(vocabSize: number, seqLen: number, dModel: number, numLayers: number, numHeads: number, dff: number) {
        super();
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.numLayers = numLayers;
        this.seqLen = seqLen;
        this.dff = dff;
        this.numHeads = numHeads;
        
        this.embeddingLayer = new Embedding(vocabSize, dModel);
        this.posEmbeddingLayer = new PositionalEmbedding(seqLen, dModel);
        this.transformerBlocks = Array.from({ length: numLayers }, () => new TransformerEncoderBlock(dModel, numHeads, dff));
        this.outputLayer = new Linear(dModel, vocabSize);
    }
    
    forward(input: Tensor): { outputLogits: Tensor } {
        let x = this.embeddingLayer.forward(input); // [batch, seqLen, dModel]
        x = x.add(this.posEmbeddingLayer.forward(x)); // Add positional encoding
        
        for(const block of this.transformerBlocks) {
            x = block.forward(x);
        }

        const outputLogits = this.outputLayer.forward(x); // [batch, seqLen, vocabSize]
        return { outputLogits };
    }

    getParameters(): Tensor[] {
        return [
            ...this.embeddingLayer.getParameters(),
            ...this.transformerBlocks.flatMap(block => block.getParameters()),
            ...this.outputLayer.getParameters()
        ];
    }
    
    getLayers(): { [key: string]: Layer | Layer[] } {
        return {
            'Embedding': this.embeddingLayer,
            'PositionalEmbedding': this.posEmbeddingLayer,
            'TransformerBlock': this.transformerBlocks,
            'Output': this.outputLayer,
        }
    }

    getArchitecture() {
      return {
          type: 'transformer',
          vocabSize: this.vocabSize,
          seqLen: this.seqLen,
          dModel: this.dModel,
          numLayers: this.numLayers,
          numHeads: this.numHeads,
          dff: this.dff
      };
    }
}

export class FlowNetModel extends BaseModelClass {
    type: 'flownet' = 'flownet';
    embeddingLayer: Embedding;
    flownetBlocks: FlowNetBlock[];
    outputLayer: Linear;

    public vocabSize: number;
    public embeddingDim: number;
    public numLayers: number;
    public seqLen: number;

    constructor(vocabSize: number, seqLen: number, embeddingDim: number, numLayers: number) {
        super();
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.numLayers = numLayers;
        this.seqLen = seqLen;
        
        this.embeddingLayer = new Embedding(vocabSize, embeddingDim);
        this.flownetBlocks = Array.from({ length: numLayers }, () => new FlowNetBlock(embeddingDim));
        this.outputLayer = new Linear(embeddingDim, vocabSize);
    }
    
    forward(input: Tensor): { outputLogits: Tensor } {
        const [batchSize, seqLen] = input.shape;
        let x = this.embeddingLayer.forward(input); // [B, S, D]

        // Initialize states for all layers
        let states = this.flownetBlocks.map(() => Tensor.zeros([batchSize, this.embeddingDim]));
        
        const outputs: Tensor[] = [];
        // Process sequence step-by-step
        for (let t = 0; t < seqLen; t++) {
            let stepInput = x.slice([0, t, 0], [batchSize, 1, this.embeddingDim]).reshape([batchSize, this.embeddingDim]);

            for(let l = 0; l < this.numLayers; l++) {
                const { output, newState } = this.flownetBlocks[l].forward(stepInput, states[l]);
                stepInput = output;
                states[l] = newState.detach(); // Detach to prevent gradients from flowing endlessly through time
            }
            outputs.push(stepInput.reshape([batchSize, 1, this.embeddingDim]));
        }

        const stackedOutputs = Tensor.concat(outputs, 1); // Concat along sequence dimension
        const outputLogits = this.outputLayer.forward(stackedOutputs); // [B, S, V]
        return { outputLogits };
    }

    getParameters(): Tensor[] {
        return [
            ...this.embeddingLayer.getParameters(),
            ...this.flownetBlocks.flatMap(block => block.getParameters()),
            ...this.outputLayer.getParameters()
        ];
    }
    
    getLayers(): { [key: string]: Layer | Layer[] } {
        return {
            'Embedding': this.embeddingLayer,
            'FlowNetBlock': this.flownetBlocks,
            'Output': this.outputLayer,
        }
    }

    getArchitecture() {
      return {
          type: 'flownet',
          vocabSize: this.vocabSize,
          seqLen: this.seqLen,
          embeddingDim: this.embeddingDim,
          numLayers: this.numLayers,
      };
    }
}


export type AnyModel = WordWiseModel | TransformerModel | FlowNetModel;


export function serializeModel(model: AnyModel, vocabData: VocabData): string {
    const layersData: { [key: string]: any } = {};

    Object.entries(model.getLayers()).forEach(([layerName, layerOrLayers]) => {
        const layers = Array.isArray(layerOrLayers) ? layerOrLayers : [layerOrLayers];
        const serializedLayers = layers.map(layer => {
            const layerParams: { [key: string]: { data: number[], shape: number[] } } = {};
             layer.getParameters().forEach(param => {
                if (!param.name) {
                    console.warn(`Parameter in layer ${layerName} is missing a name and will not be serialized.`);
                    return;
                }
                layerParams[param.name] = {
                    data: Array.from(param.data),
                    shape: param.shape
                };
            });
            return layerParams;
        });
        
       layersData[layerName] = layers.length === 1 ? serializedLayers[0] : serializedLayers;
    });
    
    const architecture = model.getArchitecture();
    let vocabInfo: any = {};

    if ('vocab' in vocabData) {
        vocabInfo = { vocab: vocabData.vocab };
    } else if ('labels' in vocabData) {
        vocabInfo = { labels: vocabData.labels };
    }


    const dataToSave = {
        architecture,
        weights: layersData,
        ...vocabInfo
    };

    return JSON.stringify(dataToSave);
}


export function deserializeModel(jsonString: string): { model: AnyModel, vocabData: VocabData } {
    const savedData = JSON.parse(jsonString);

    if (!savedData.architecture || !savedData.weights) {
        throw new Error("Invalid model file format: missing architecture or weights.");
    }
    
    const { architecture, weights } = savedData;
    let model: AnyModel;
    let vocabData: VocabData;

    if (architecture.type === 'lstm' || architecture.type === 'transformer' || architecture.type === 'flownet') {
        if (!savedData.vocab) throw new Error("Missing vocab for text model.");
        const vocab = savedData.vocab;
        const wordToIndex = new Map(vocab.map((word: string, i: number) => [word, i]));
        const indexToWord = new Map(vocab.map((word: string, i: number) => [i, word]));
        const vocabSize = vocab.length;
        vocabData = { vocab, wordToIndex, indexToWord, vocabSize };
        
        if (vocabSize !== architecture.vocabSize) {
             console.warn(`Vocabulary size mismatch. Loaded model has ${architecture.vocabSize}, but current vocab is ${vocabSize}. This might be OK if you are fine-tuning.`);
        }
        
        if (architecture.type === 'lstm') {
            const { embeddingDim, hiddenSize } = architecture;
            model = new WordWiseModel(architecture.vocabSize, embeddingDim, hiddenSize);
        } else if (architecture.type === 'transformer') {
            const { seqLen, dModel, numLayers, numHeads, dff } = architecture;
            model = new TransformerModel(architecture.vocabSize, seqLen, dModel, numLayers, numHeads, dff);
        } else { // flownet
            const { seqLen, embeddingDim, numLayers } = architecture;
            model = new FlowNetModel(architecture.vocabSize, seqLen, embeddingDim, numLayers);
        }
    } else {
        throw new Error(`Unknown model type in saved file: ${architecture.type}`);
    }

    Object.entries(model.getLayers()).forEach(([layerName, layerOrLayers]) => {
        const savedLayerWeights = weights[layerName];
        if (!savedLayerWeights) throw new Error(`Weights for layer ${layerName} not found in file.`);
        
        const layers = Array.isArray(layerOrLayers) ? layerOrLayers : [layerOrLayers];
        const savedWeightsArray = Array.isArray(savedLayerWeights) ? savedLayerWeights : [savedLayerWeights];

        if (layers.length !== savedWeightsArray.length) throw new Error(`Layer count mismatch for ${layerName}`);

        layers.forEach((layer, index) => {
            const savedLayerParams = savedWeightsArray[index];
            layer.getParameters().forEach(param => {
                if (!param.name) return;
                const savedParam = savedLayerParams[param.name];
                if (!savedParam) throw new Error(`Parameter ${param.name} for layer ${layerName}[${index}] not found.`);
                
                const expectedSize = param.size;
                const loadedSize = savedParam.data.length;

                if (expectedSize !== loadedSize) {
                    throw new Error(`Weight size mismatch for ${param.name} in layer ${layerName}_${index}. Expected ${expectedSize}, got ${loadedSize}.`);
                }
                param.data.set(savedParam.data);
            });
        });
    });

    return { model, vocabData };
}

    
