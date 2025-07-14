// src/lib/model.ts
import { Embedding, Linear, LSTMCell, Layer, Conv2d, Flatten, relu } from './layers';
import { Tensor } from './tensor';

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

export type BaseModel = WordWiseModel | ImageWiseModel;

/**
 * WordWiseModel: Простая рекуррентная нейронная сеть для обработки текста,
 * использующая Embedding слой, LSTM ячейку и линейный выходной слой.
 * Предназначена для задач предсказания следующего слова.
 */
export class WordWiseModel {
  type: 'text' = 'text';
  embeddingLayer: Embedding; // Слой для преобразования индексов слов в векторы
  lstmCell: LSTMCell;         // Ячейка LSTM для обработки последовательностей
  outputLayer: Linear;        // Выходной слой для предсказания следующего слова
  public hiddenSize: number;          // Размер скрытого состояния LSTM
  public vocabSize: number;
  public embeddingDim: number;

  /**
   * @param vocabSize Размер словаря (количество уникальных слов).
   * @param embeddingDim Размерность векторов эмбеддингов слов.
   * @param hiddenSize Размерность скрытого состояния LSTM.
   */
  constructor(vocabSize: number, embeddingDim: number, hiddenSize: number) {
    this.embeddingLayer = new Embedding(vocabSize, embeddingDim);
    this.lstmCell = new LSTMCell(embeddingDim, hiddenSize);
    this.outputLayer = new Linear(hiddenSize, vocabSize);
    this.hiddenSize = hiddenSize;
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
  }

  /**
   * Выполняет один шаг прямого прохода модели для обработки одного слова в батче.
   * @param input Тензор входных индексов слов: [batchSize, 1].
   * @param prevH Предыдущее скрытое состояние LSTM: [batchSize, hiddenSize].
   * @param prevC Предыдущее состояние ячейки LSTM: [batchSize, hiddenSize].
   * @returns Объект, содержащий:
   * - `outputLogits`: Сырые логиты для предсказания следующего слова: [batchSize, vocabSize].
   * - `h`: Новое скрытое состояние LSTM: [batchSize, hiddenSize].
   * - `c`: Новое состояние ячейки LSTM: [batchSize, hiddenSize].
   */
  forward(input: Tensor, prevH?: Tensor, prevC?: Tensor): { outputLogits: Tensor, h: Tensor, c: Tensor } {
    if (!prevH || !prevC) {
      throw new Error("Missing previous hidden/cell state for WordWiseModel forward pass");
    }
    // 1. Embedding Layer: преобразует индексы слов в плотные векторы
    const embeddedInput = this.embeddingLayer.forward(input); // [batchSize, embeddingDim]

    // 2. LSTM Cell: обрабатывает текущий вход и предыдущие состояния
    const { h, c } = this.lstmCell.forward(embeddedInput, prevH, prevC); // h: [batchSize, hiddenSize], c: [batchSize, hiddenSize]

    // 3. Output Layer: преобразует скрытое состояние в логиты для предсказания следующего слова
    const outputLogits = this.outputLayer.forward(h); // [batchSize, vocabSize]

    return { outputLogits, h, c };
  }

  /**
   * Получает все обучаемые параметры (веса и смещения) из всех слоев модели.
   * @returns Массив тензоров-параметров.
   */
  getParameters(): Tensor[] {
    return [
      ...this.embeddingLayer.getParameters(),
      ...this.lstmCell.getParameters(),
      ...this.outputLayer.getParameters(),
    ];
  }
  
  getLayers(): { [key: string]: Layer } {
    return {
      embeddingLayer: this.embeddingLayer,
      lstmCell: this.lstmCell,
      outputLayer: this.outputLayer,
    }
  }

  /**
   * Инициализирует начальные скрытое состояние (h0) и состояние ячейки (c0)
   * для начала новой последовательности.
   * @param batchSize Размер батча.
   * @returns Объект с начальными состояниями.
   */
  initializeStates(batchSize: number): { h0: Tensor, c0: Tensor } {
    // Инициализируем нулями
    const h0 = Tensor.zeros([batchSize, this.hiddenSize]);
    const c0 = Tensor.zeros([batchSize, this.hiddenSize]);
    return { h0, c0 };
  }
}

export class ImageWiseModel {
    type: 'image' = 'image';
    // A very simple CNN: Conv -> ReLU -> Flatten -> Linear
    conv1: Conv2d;
    flatten: Flatten;
    linear1: Linear;
    public numClasses: number;

    constructor(numClasses: number, imageWidth: number, imageHeight: number, inChannels: number) {
        this.numClasses = numClasses;

        // Note: This is a simplified architecture. A real one would have more layers.
        // The output size of Conv2d and Flatten would need to be calculated precisely.
        this.conv1 = new Conv2d(inChannels, 8, 3, 1, 1); // outChannels=8, kernel=3x3
        this.flatten = new Flatten();
        
        // This is a simplification. The real input size to the linear layer
        // depends on the output of the conv layer.
        const flattenedSize = 8 * imageWidth * imageHeight;
        this.linear1 = new Linear(flattenedSize, numClasses);
    }
    
    forward(input: Tensor): { outputLogits: Tensor } {
        // Input shape: [batch, channels, height, width]
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
            conv1: this.conv1,
            flatten: this.flatten,
            linear1: this.linear1,
        }
    }
}

// Функции для сохранения и загрузки модели

/**
 * Сериализует модель и данные словаря в JSON-строку.
 * @param model Экземпляр WordWiseModel.
 * @param vocabData Данные словаря.
 * @returns JSON-строка.
 */
export function serializeModel(model: BaseModel, vocabData: VocabData): string {
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

    if (model.type === 'text' && 'vocab' in vocabData) {
        architecture = { ...architecture, vocabSize: model.vocabSize, embeddingDim: model.embeddingDim, hiddenSize: model.hiddenSize };
        vocabInfo = { vocab: vocabData.vocab };
    } else if (model.type === 'image' && 'labels' in vocabData) {
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

/**
 * Десериализует модель из JSON-строки.
 * @param jsonString JSON-строка с данными модели.
 * @returns Объект с экземпляром WordWiseModel и данными словаря.
 */
export function deserializeModel(jsonString: string): { model: BaseModel, vocabData: VocabData } {
    const savedData = JSON.parse(jsonString);

    if (!savedData.architecture || !savedData.weights) {
        throw new Error("Invalid model file format: missing architecture or weights.");
    }
    
    const { architecture, weights } = savedData;
    let model: BaseModel;
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
        // NOTE: We're passing dummy dimensions here. In a real scenario, this should be saved in the model JSON.
        model = new ImageWiseModel(numClasses, 32, 32, 3);
    } else {
        throw new Error(`Unknown model type in saved file: ${architecture.type}`);
    }


    // Загружаем веса
    Object.entries(model.getLayers()).forEach(([layerName, layer]) => {
        const savedLayerWeights = weights[layerName];
        if (!savedLayerWeights) throw new Error(`Weights for layer ${layerName} not found in file.`);
        
        layer.getParameters().forEach(param => {
            if (!param.name) return; // Skip unnamed params
            const savedParam = savedLayerWeights[param.name];
            if (!savedParam) throw new Error(`Parameter ${param.name} for layer ${layerName} not found.`);
            if (param.size !== savedParam.data.length) throw new Error(`Weight size mismatch for ${param.name}.`);

            param.data.set(savedParam.data); // Загружаем веса в существующий тензор
        });
    });

    return { model, vocabData };
}
