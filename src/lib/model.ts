// src/lib/model.ts
import { Embedding, Linear, LSTMCell, Layer } from './layers';
import { Tensor } from './tensor';

type VocabDataType = {
  vocab: string[];
  wordToIndex: Map<string, number>;
  indexToWord: Map<number, string>;
  vocabSize: number;
};

/**
 * WordWiseModel: Простая рекуррентная нейронная сеть для обработки текста,
 * использующая Embedding слой, LSTM ячейку и линейный выходной слой.
 * Предназначена для задач предсказания следующего слова.
 */
export class WordWiseModel {
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
  forwardStep(input: Tensor, prevH: Tensor, prevC: Tensor): { outputLogits: Tensor, h: Tensor, c: Tensor } {
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

// Функции для сохранения и загрузки модели

/**
 * Сериализует модель и данные словаря в JSON-строку.
 * @param model Экземпляр WordWiseModel.
 * @param vocabData Данные словаря.
 * @returns JSON-строка.
 */
export function serializeModel(model: WordWiseModel, vocabData: VocabDataType): string {
    const layersData: { [key: string]: { [key: string]: { data: number[], shape: number[] } } } = {};

    Object.entries(model.getLayers()).forEach(([layerName, layer]) => {
        layersData[layerName] = {};
        layer.getParameters().forEach(param => {
            layersData[layerName][param.name] = {
                data: Array.from(param.data),
                shape: param.shape
            };
        });
    });

    const dataToSave = {
        architecture: {
            vocabSize: model.vocabSize,
            embeddingDim: model.embeddingDim,
            hiddenSize: model.hiddenSize
        },
        weights: layersData,
        vocab: vocabData.vocab
    };

    return JSON.stringify(dataToSave, null, 2);
}

/**
 * Десериализует модель из JSON-строки.
 * @param jsonString JSON-строка с данными модели.
 * @returns Объект с экземпляром WordWiseModel и данными словаря.
 */
export function deserializeModel(jsonString: string): { model: WordWiseModel, vocabData: VocabDataType } {
    const savedData = JSON.parse(jsonString);

    if (!savedData.architecture || !savedData.weights || !savedData.vocab) {
        throw new Error("Invalid model file format.");
    }
    
    // Восстанавливаем словарь
    const vocab = savedData.vocab;
    const wordToIndex = new Map(vocab.map((word: string, i: number) => [word, i]));
    const indexToWord = new Map(vocab.map((word: string, i: number) => [i, word]));
    const vocabSize = vocab.length;
    const vocabData = { vocab, wordToIndex, indexToWord, vocabSize };
    
    // Проверяем соответствие архитектуры
    if (vocabSize !== savedData.architecture.vocabSize) {
        throw new Error("Vocabulary size mismatch between loaded model and its architecture description.");
    }
    
    // Создаем новую модель с той же архитектурой
    const { embeddingDim, hiddenSize } = savedData.architecture;
    const model = new WordWiseModel(vocabSize, embeddingDim, hiddenSize);

    // Загружаем веса
    Object.entries(model.getLayers()).forEach(([layerName, layer]) => {
        const savedLayerWeights = savedData.weights[layerName];
        if (!savedLayerWeights) throw new Error(`Weights for layer ${layerName} not found in file.`);
        
        layer.getParameters().forEach(param => {
            const savedParam = savedLayerWeights[param.name];
            if (!savedParam) throw new Error(`Parameter ${param.name} for layer ${layerName} not found.`);
            if (param.size !== savedParam.data.length) throw new Error(`Weight size mismatch for ${param.name}.`);

            param.data.set(savedParam.data); // Загружаем веса в существующий тензор
        });
    });

    return { model, vocabData };
}