// src/lib/model.ts
import { Embedding, Linear, LSTMCell, softmax } from './layers';
import { Tensor } from './tensor';

/**
 * WordWiseModel: Простая рекуррентная нейронная сеть для обработки текста,
 * использующая Embedding слой, LSTM ячейку и линейный выходной слой.
 * Предназначена для задач предсказания следующего слова.
 */
export class WordWiseModel {
  private embeddingLayer: Embedding; // Слой для преобразования индексов слов в векторы
  private lstmCell: LSTMCell;         // Ячейка LSTM для обработки последовательностей
  private outputLayer: Linear;        // Выходной слой для предсказания следующего слова
  public hiddenSize: number;          // Размер скрытого состояния LSTM

  /**
   * @param vocabSize Размер словаря (количество уникальных слов).
   * @param embeddingDim Размерность векторов эмбеддингов слов.
   * @param hiddenSize Размерность скрытого состояния LSTM.
   */
  constructor(vocabSize: number, embeddingDim: number, hiddenSize: number) {
    this.embeddingLayer = new Embedding(vocabSize, embeddingDim);
    this.lstmCell = new LSTMCell(embeddingDim, hiddenSize);
    this.outputLayer = new Linear(hiddenSize, vocabSize);
    this.hiddenSize = hiddenSize; // Сохраняем размер скрытого состояния для инициализации
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
