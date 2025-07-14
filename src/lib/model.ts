// src/lib/model.ts
import { Embedding, Linear, LSTMCell, Layer, Conv2d, Flatten, relu, crossEntropyLossWithSoftmaxGrad } from './layers';
import { SGD } from './optimizer';
import { Tensor } from './tensor';
import { createTextBatches, createImageBatches } from '@/utils/batching';


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
    abstract type: 'text' | 'image';

    abstract getParameters(): Tensor[];
    abstract getLayers(): { [key: string]: Layer };

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

        const batches = this.type === 'text'
            ? createTextBatches(inputs, targets, options.batchSize)
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
                } else if (this instanceof ImageWiseModel) {
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
                const gradientInfo = Object.entries(this.getLayers()).map(([layerName, layer]) => {
                    let totalGradMag = 0;
                    let paramCount = 0;
                    layer.getParameters().forEach(p => {
                        if (p.grad) {
                           totalGradMag += p.grad.data.reduce((acc, val) => acc + Math.abs(val), 0) / p.grad.size;
                        }
                    });
                     return { layer: layerName, avgGrad: totalGradMag };
                });

                callbacks.onEpochEnd({
                    epoch: currentEpoch,
                    loss: avgEpochLoss,
                    gradients: gradientInfo
                });
            }
            // Give the main thread a chance to breathe
            await new Promise(resolve => setTimeout(resolve, 10));
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
    const { h, c } = this.lstmCell.forward(embeddedInput, prevH, prevC);
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

export class ImageWiseModel extends BaseModelClass {
    type: 'image' = 'image';
    conv1: Conv2d;
    flatten: Flatten;
    linear1: Linear;
    public numClasses: number;

    constructor(numClasses: number, imageWidth: number, imageHeight: number, inChannels: number) {
        super();
        this.numClasses = numClasses;
        this.conv1 = new Conv2d(inChannels, 8, 3, 1, 1);
        this.flatten = new Flatten();
        const flattenedSize = 8 * imageWidth * imageHeight;
        this.linear1 = new Linear(flattenedSize, numClasses);
    }
    
    forward(input: Tensor): { outputLogits: Tensor } {
        let x = this.conv1.forward(input);
        x = relu(x);
        x = this.flatten.forward(x);
        const outputLogits = this.linear1.forward(x);
        return { outputLogits };
    }

    getParameters(): Tensor[] {
        return [
            ...this.conv1.getParameters(),
            ...this.linear1.getParameters()
        ];
    }
    
    getLayers(): { [key: string]: Layer } {
        return {
            'Convolutional': this.conv1,
            'Output': this.linear1,
        }
    }
}

export type AnyModel = WordWiseModel | ImageWiseModel;


export function serializeModel(model: AnyModel, vocabData: VocabData): string {
    const layersData: { [key: string]: { [key: string]: { data: number[], shape: number[] } } } = {};

    Object.entries(model.getLayers()).forEach(([layerName, layer]) => {
        layersData[layerName] = {};
        layer.getParameters().forEach(param => {
            if (!param.name) {
                console.warn(`Parameter in layer ${layerName} is missing a name and will not be serialized.`);
                return;
            }
            layersData[layerName][param.name] = {
                data: Array.from(param.data),
                shape: param.shape
            };
        });
    });
    
    let architecture: any = { type: model.type };
    let vocabInfo: any = {};

    if (model.type === 'text' && model instanceof WordWiseModel && 'vocab' in vocabData) {
        architecture = { ...architecture, vocabSize: model.vocabSize, embeddingDim: model.embeddingDim, hiddenSize: model.hiddenSize };
        vocabInfo = { vocab: vocabData.vocab };
    } else if (model.type === 'image' && model instanceof ImageWiseModel && 'labels' in vocabData) {
        architecture = { ...architecture, numClasses: model.numClasses };
        vocabInfo = { labels: vocabData.labels };
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

    if (architecture.type === 'text') {
        if (!savedData.vocab) throw new Error("Missing vocab for text model.");
        const vocab = savedData.vocab;
        const wordToIndex = new Map(vocab.map((word: string, i: number) => [word, i]));
        const indexToWord = new Map(vocab.map((word: string, i: number) => [i, word]));
        const vocabSize = vocab.length;
        vocabData = { vocab, wordToIndex, indexToWord, vocabSize };
        
        if (vocabSize !== architecture.vocabSize) {
             throw new Error("Vocabulary size mismatch between loaded model and its architecture description.");
        }
        
        const { embeddingDim, hiddenSize } = architecture;
        model = new WordWiseModel(vocabSize, embeddingDim, hiddenSize);
    } else if (architecture.type === 'image') {
        if (!savedData.labels) throw new Error("Missing labels for image model.");
        const labels = savedData.labels;
        const labelToIndex = new Map(labels.map((label: string, i: number) => [label, i]));
        const indexToLabel = new Map(labels.map((label: string, i: number) => [i, label]));
        const numClasses = labels.length;
        vocabData = { labels, labelToIndex, indexToLabel, numClasses };

        if (numClasses !== architecture.numClasses) {
            throw new Error("Number of classes mismatch between loaded model and its architecture description.");
        }
        model = new ImageWiseModel(numClasses, 32, 32, 3);
    } else {
        throw new Error(`Unknown model type in saved file: ${architecture.type}`);
    }

    Object.entries(model.getLayers()).forEach(([layerName, layer]) => {
        const savedLayerWeights = weights[layerName];
        if (!savedLayerWeights) throw new Error(`Weights for layer ${layerName} not found in file.`);
        
        layer.getParameters().forEach(param => {
            if (!param.name) return;
            const savedParam = savedLayerWeights[param.name];
            if (!savedParam) throw new Error(`Parameter ${param.name} for layer ${layerName} not found.`);
            if (param.size !== savedParam.data.length) throw new Error(`Weight size mismatch for ${param.name}.`);

            param.data.set(savedParam.data);
        });
    });

    return { model, vocabData };
}
