// src/lib/model.ts
import { Embedding, Linear, LSTMCell, Layer, Flatten, crossEntropyLossWithSoftmaxGrad, PositionalEmbedding } from './layers';
import { TransformerEncoderBlock } from './transformer';
import { SGD } from './optimizer';
import { Tensor } from './tensor';
import { createSequenceBatches, createImageBatches } from '@/utils/batching';


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
    onEpochEnd?: (log: { epoch: number; loss: number; gradients: { layer: string; avgGrad: number }[] }) => void;
}

// --- Base Model Class ---
abstract class BaseModelClass {
    abstract type: 'text' | 'image' | 'transformer';

    abstract getParameters(): Tensor[];
    abstract getLayers(): { [key: string]: Layer | Layer[] };

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

        const batches = this.type === 'transformer'
            ? createSequenceBatches(inputs, targets, options.batchSize, (this as TransformerModel).seqLen)
            : createImageBatches(inputs, targets, options.batchSize);

        if (batches.length === 0) {
            throw new Error("Could not create batches. Check your data.");
        }

        for (let epoch = 0; epoch < options.epochs; epoch++) {
            let epochLoss = 0;

            for (const batch of batches) {
                const batchInputs = new Tensor(batch.inputs.data, batch.inputs.shape);
                const batchTargets = new Tensor(batch.targets.data, batch.targets.shape);

                let predictionLogits;
                 if (this instanceof WordWiseModel) {
                    let {h0: h, c0: c} = this.initializeStates(batchInputs.shape[0]);
                    predictionLogits = this.forward(batchInputs, h, c).outputLogits;
                } else if (this instanceof TransformerModel) {
                    predictionLogits = this.forward(batchInputs).outputLogits;
                } else {
                    throw new Error("Unknown model instance type in fit method");
                }

                const loss = crossEntropyLossWithSoftmaxGrad(predictionLogits, batchTargets);
                if(loss.data.length === 1) {
                    epochLoss += loss.data[0];
                }

                loss.backward();
                optimizer.step(this.getParameters());
            }

            const avgEpochLoss = epochLoss / batches.length;
            const currentEpoch = startEpoch + epoch;

            // --- Gradient Visualization Callback ---
            if (callbacks?.onEpochEnd) {
                const gradientInfo = Object.entries(this.getLayers()).flatMap(([layerName, layerOrLayers]) => {
                    const layers = Array.isArray(layerOrLayers) ? layerOrLayers : [layerOrLayers];
                    return layers.map((layer, index) => {
                         let totalGradMag = 0;
                         layer.getParameters().forEach(p => {
                            if (p.grad) {
                               totalGradMag += p.grad.data.reduce((acc, val) => acc + Math.abs(val), 0) / p.grad.size;
                            }
                         });
                         const finalLayerName = layers.length > 1 ? `${layerName}_${index}` : layerName;
                         return { layer: finalLayerName, avgGrad: totalGradMag };
                    });
                });

                callbacks.onEpochEnd({
                    epoch: currentEpoch,
                    loss: avgEpochLoss,
                    gradients: gradientInfo
                });
            }
             // Give the main thread a chance to breathe
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }
}


/**
 * WordWiseModel: Простая рекуррентная нейронная сеть для обработки текста,
 * использующая Embedding слой, LSTM ячейку и линейный выходной слой.
 * Предназначена для задач предсказания следующего слова.
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

  forward(input: Tensor, prevH: Tensor, prevC: Tensor): { outputLogits: Tensor, h: Tensor, c: Tensor } {
    const embeddedInput = this.embeddingLayer.forward(input);
    // Note: For simplicity, this LSTM model processes sequences as a batch of independent items.
    // A true sequence-to-sequence model would iterate through the sequence dimension.
    const { h, c } = this.lstmCell.forward(embeddedInput.reshape([-1, this.embeddingDim]), prevH, prevC);
    const outputLogits = this.outputLayer.forward(h);
    return { outputLogits, h, c };
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
        x = this.posEmbeddingLayer.forward(x);     // [batch, seqLen, dModel]

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
}


export type AnyModel = WordWiseModel | TransformerModel;


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
    
    let architecture: any;
    let vocabInfo: any = {};

    if (model instanceof WordWiseModel && 'vocab' in vocabData) {
        architecture = { type: 'text', vocabSize: model.vocabSize, embeddingDim: model.embeddingDim, hiddenSize: model.hiddenSize };
        vocabInfo = { vocab: vocabData.vocab };
    } else if (model instanceof TransformerModel && 'vocab' in vocabData) {
        architecture = {
            type: 'transformer',
            vocabSize: model.vocabSize,
            seqLen: model.seqLen,
            dModel: model.dModel,
            numLayers: model.numLayers,
            numHeads: model.numHeads,
            dff: model.dff
        };
        vocabInfo = { vocab: vocabData.vocab };
    } else {
        throw new Error("Unsupported model or vocab data type for serialization.");
    }

    const dataToSave = {
        architecture,
        weights: layersData,
        ...vocabInfo
    };

    return JSON.stringify(dataToSave, null, 2);
}


export function deserializeModel(jsonString: string): { model: AnyModel, vocabData: VocabData } {
    const savedData = JSON.parse(jsonString);

    if (!savedData.architecture || !savedData.weights) {
        throw new Error("Invalid model file format: missing architecture or weights.");
    }
    
    const { architecture, weights } = savedData;
    let model: AnyModel;
    let vocabData: VocabData;

    if (architecture.type === 'text' || architecture.type === 'transformer') {
         if (!savedData.vocab) throw new Error("Missing vocab for text model.");
        const vocab = savedData.vocab;
        const wordToIndex = new Map(vocab.map((word: string, i: number) => [word, i]));
        const indexToWord = new Map(vocab.map((word: string, i: number) => [i, word]));
        const vocabSize = vocab.length;
        vocabData = { vocab, wordToIndex, indexToWord, vocabSize };
        
        if (vocabSize !== architecture.vocabSize) {
             throw new Error("Vocabulary size mismatch between loaded model and its architecture description.");
        }
        
        if (architecture.type === 'text') {
            const { embeddingDim, hiddenSize } = architecture;
            model = new WordWiseModel(vocabSize, embeddingDim, hiddenSize);
        } else { // transformer
            const { seqLen, dModel, numLayers, numHeads, dff } = architecture;
            model = new TransformerModel(vocabSize, seqLen, dModel, numLayers, numHeads, dff);
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
                if (!savedParam) throw new Error(`Parameter ${param.name} for layer ${layerName} not found.`);
                if (param.size !== savedParam.data.length) {
                    throw new Error(`Weight size mismatch for ${param.name} in layer ${layerName}. Expected ${param.size}, got ${savedParam.data.length}.`);
                }
                param.data.set(savedParam.data);
            });
        });
    });

    return { model, vocabData };
}
