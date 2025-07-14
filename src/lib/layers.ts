// src/lib/layers.ts
import { Tensor } from './tensor';

/**
 * Базовый абстрактный класс для всех слоев нейронной сети.
 * Определяет общий интерфейс для получения параметров и выполнения прямого прохода.
 */
export abstract class Layer {
  abstract parameters: Tensor[]; // Обучаемые параметры слоя
  // forward может принимать несколько входных тензоров и возвращать несколько
  abstract forward(...inputs: Tensor[]): Tensor | { [key: string]: Tensor };

  /**
   * Возвращает все обучаемые параметры слоя.
   * @returns Массив тензоров-параметров.
   */
  getParameters(): Tensor[] {
    return this.parameters;
  }
}

// --- Слой Embedding ---

/**
 * Слой Embedding: преобразует дискретные индексы слов в плотные векторы (эмбеддинги).
 * По сути, это обучаемая таблица поиска.
 */
export class Embedding extends Layer {
  weights: Tensor; // Таблица эмбеддингов: [vocabSize, embeddingDim]
  parameters: Tensor[];

  /**
   * @param vocabSize Размер словаря (количество уникальных слов).
   * @param embeddingDim Размерность вектора эмбеддинга для каждого слова.
   */
  constructor(vocabSize: number, embeddingDim: number) {
    super();
    // Инициализация весов: случайные значения из нормального распределения
    const limit = Math.sqrt(1 / embeddingDim); // Эвристика для эмбеддингов
    this.weights = Tensor.randn([vocabSize, embeddingDim], limit);
    this.parameters = [this.weights];
  }

  /**
   * Прямой проход слоя Embedding.
   * @param input Тензор с индексами слов: [batchSize, 1] или [batchSize].
   * @returns Тензор эмбеддингов: [batchSize, embeddingDim].
   */
  forward(input: Tensor): Tensor {
    if (input.shape.length > 2 || (input.shape.length === 2 && input.shape[1] !== 1)) {
        throw new Error("Embedding layer input must be a 1D tensor of indices ([batchSize]) or 2D ([batchSize, 1]).");
    }
    const batchSize = input.shape[0];
    const embeddingDim = this.weights.shape[1];
    const vocabSize = this.weights.shape[0];

    const resultData = new Float32Array(batchSize * embeddingDim);

    const parentTensor = this.weights; // Обучаемый тензор, к которому будут применяться градиенты

    // Для каждого индекса в батче, выбираем соответствующую строку из таблицы эмбеддингов
    for (let i = 0; i < batchSize; i++) {
      const wordIndex = Math.floor(input.data[i]);
      if (wordIndex < 0 || wordIndex >= vocabSize) {
        // Если индекс выходит за пределы, используем <unk> (индекс 0)
        console.warn(`Embedding: Word index ${wordIndex} out of vocabulary bounds (0-${vocabSize-1}). Using <unk> token.`);
        // Можно использовать <unk> токен, если он был добавлен при создании словаря
        const unkIndex = 0; // Предполагаем, что <unk> имеет индекс 0
        const embeddingRowStart = unkIndex * embeddingDim;
        const resultRowStart = i * embeddingDim;
        for (let j = 0; j < embeddingDim; j++) {
          resultData[resultRowStart + j] = this.weights.data[embeddingRowStart + j];
        }
      } else {
        const embeddingRowStart = wordIndex * embeddingDim;
        const resultRowStart = i * embeddingDim;
        for (let j = 0; j < embeddingDim; j++) {
          resultData[resultRowStart + j] = this.weights.data[embeddingRowStart + j];
        }
      }
    }

    const result = new Tensor(resultData, [batchSize, embeddingDim]);

    // Градиент для Embedding слоя:
    // Градиент от `result` должен быть добавлен к соответствующим строкам `this.weights`
    result._parents.push({
      tensor: parentTensor,
      gradFn: (grad) => {
        const dWeights = Tensor.zeros(parentTensor.shape); // Создаем тензор для агрегации градиентов
        for (let i = 0; i < batchSize; i++) {
          const wordIndex = Math.floor(input.data[i]); // Оригинальный индекс слова
          const gradRowStart = i * embeddingDim;
          const dWeightsRowStart = wordIndex * embeddingDim;
          for (let j = 0; j < embeddingDim; j++) {
            // Аккумулируем градиент, так как одно и то же слово может появиться несколько раз в батче
            dWeights.data[dWeightsRowStart + j] += grad.data[gradRowStart + j];
          }
        }
        return dWeights;
      }
    });
    return result;
  }
}


// --- Полносвязный слой (Dense / Linear) ---

/**
 * Полносвязный (линейный, Dense) слой: выполняет матричное умножение входных данных
 * на матрицу весов и добавляет вектор смещения.
 */
export class Linear extends Layer {
  weights: Tensor; // Матрица весов [inputSize, outputSize]
  bias: Tensor;    // Вектор смещения [1, outputSize] (бродкастится на весь батч)
  parameters: Tensor[];

  /**
   * @param inputSize Размерность входных признаков.
   * @param outputSize Размерность выходных признаков.
   */
  constructor(inputSize: number, outputSize: number) {
    super();
    // Инициализация весов методом Glorot/Xavier Uniform
    const limit = Math.sqrt(6 / (inputSize + outputSize));
    this.weights = Tensor.randn([inputSize, outputSize], limit);
    this.bias = Tensor.zeros([1, outputSize]); // Смещения обычно инициализируются нулями

    this.parameters = [this.weights, this.bias];
  }

  /**
   * Прямой проход линейного слоя.
   * @param input Входной тензор: [batchSize, inputSize].
   * @returns Выходной тензор: [batchSize, outputSize].
   */
  forward(input: Tensor): Tensor {
    // Z = X @ W + B
    const matMulResult = input.dot(this.weights); // [batchSize, outputSize]
    const output = matMulResult.add(this.bias);   // Бродкастинг bias: [batchSize, outputSize]
    return output;
  }
}

// --- Функции активации ---

/**
 * Активационная функция ReLU (Rectified Linear Unit).
 * @param input Входной тензор.
 * @returns Тензор с примененной ReLU.
 */
export function relu(input: Tensor): Tensor {
  return input.apply(
    (x) => Math.max(0, x), // Функция ReLU
    (x, y, g) => x > 0 ? g : 0 // Производная ReLU: 1 если x > 0, иначе 0
  );
}

/**
 * Активационная функция Sigmoid.
 * @param input Входной тензор.
 * @returns Тензор с примененной Sigmoid.
 */
export function sigmoid(input: Tensor): Tensor {
  return input.apply(
    (x) => 1 / (1 + Math.exp(-x)), // Функция Sigmoid
    (x, y, g) => g * y * (1 - y) // Производная Sigmoid: y * (1 - y)
  );
}

/**
 * Активационная функция Tanh (Гиперболический тангенс).
 * @param input Входной тензор.
 * @returns Тензор с примененной Tanh.
 */
export function tanh(input: Tensor): Tensor {
  return input.apply(
    (x) => Math.tanh(x), // Функция Tanh
    (x, y, g) => g * (1 - y * y) // Производная Tanh: 1 - y^2
  );
}

// --- LSTM Cell (для обработки последовательностей) ---

/**
 * Упрощенная ячейка LSTM (Long Short-Term Memory).
 * Обрабатывает один временной шаг в последовательности.
 */
export class LSTMCell extends Layer {
    // Веса и смещения для каждого из четырех гейтов (input, forget, output, cell state candidate)
    Wi: Tensor; Ui: Tensor; Bi: Tensor; // Input gate parameters
    Wf: Tensor; Uf: Tensor; Bf: Tensor; // Forget gate parameters
    Wo: Tensor; Uo: Tensor; Bo: Tensor; // Output gate parameters
    Wc: Tensor; Uc: Tensor; Bc: Tensor; // Cell state candidate parameters

    parameters: Tensor[];

    /**
     * @param inputSize Размерность входного вектора (например, embeddingDim).
     * @param hiddenSize Размерность скрытого состояния (памяти) ячейки.
     */
    constructor(inputSize: number, hiddenSize: number) {
        super();
        // Инициализация весов гейтов
        const limit_x = Math.sqrt(6 / (inputSize + hiddenSize));
        const limit_h = Math.sqrt(6 / (hiddenSize + hiddenSize));

        this.Wi = Tensor.randn([inputSize, hiddenSize], limit_x);
        this.Ui = Tensor.randn([hiddenSize, hiddenSize], limit_h);
        this.Bi = Tensor.zeros([1, hiddenSize]);

        this.Wf = Tensor.randn([inputSize, hiddenSize], limit_x);
        this.Uf = Tensor.randn([hiddenSize, hiddenSize], limit_h);
        this.Bf = Tensor.zeros([1, hiddenSize]);

        this.Wo = Tensor.randn([inputSize, hiddenSize], limit_x);
        this.Uo = Tensor.randn([hiddenSize, hiddenSize], limit_h);
        this.Bo = Tensor.zeros([1, hiddenSize]);

        this.Wc = Tensor.randn([inputSize, hiddenSize], limit_x);
        this.Uc = Tensor.randn([hiddenSize, hiddenSize], limit_h);
        this.Bc = Tensor.zeros([1, hiddenSize]);

        this.parameters = [
            this.Wi, this.Ui, this.Bi,
            this.Wf, this.Uf, this.Bf,
            this.Wo, this.Uo, this.Bo,
            this.Wc, this.Uc, this.Bc
        ];
    }

    /**
     * Прямой проход ячейки LSTM для одного временного шага.
     * @param input Текущий входной тензор: [batchSize, inputSize].
     * @param prevH Предыдущее скрытое состояние: [batchSize, hiddenSize].
     * @param prevC Предыдущее состояние ячейки: [batchSize, hiddenSize].
     * @returns Объект с новым скрытым состоянием (h) и состоянием ячейки (c).
     */
    forward(input: Tensor, prevH: Tensor, prevC: Tensor): { h: Tensor, c: Tensor } {
        // Вычисления для Input Gate: i_t = sigmoid(W_xi * x_t + W_hi * h_{t-1} + b_i)
        const i_t = sigmoid(input.dot(this.Wi).add(prevH.dot(this.Ui)).add(this.Bi));
        // Вычисления для Forget Gate: f_t = sigmoid(W_xf * x_t + W_hf * h_{t-1} + b_f)
        const f_t = sigmoid(input.dot(this.Wf).add(prevH.dot(this.Uf)).add(this.Bf));
        // Вычисления для Cell State Candidate: c_tilde_t = tanh(W_xc * x_t + W_hc * h_{t-1} + b_c)
        const c_tilde_t = tanh(input.dot(this.Wc).add(prevH.dot(this.Uc)).add(this.Bc));

        // Обновление состояния ячейки: c_t = f_t * c_{t-1} + i_t * c_tilde_t
        const c_t = f_t.mul(prevC).add(i_t.mul(c_tilde_t));

        // Вычисления для Output Gate: o_t = sigmoid(W_xo * x_t + W_ho * h_{t-1} + b_o)
        const o_t = sigmoid(input.dot(this.Wo).add(prevH.dot(this.Uo)).add(this.Bo));
        // Обновление скрытого состояния: h_t = o_t * tanh(c_t)
        const h_t = o_t.mul(tanh(c_t));

        return { h: h_t, c: c_t };
    }
}

// --- Функции потерь ---

/**
 * Функция потерь Cross-Entropy (кросс-энтропия)
 * с интегрированным градиентом Softmax для численной стабильности.
 * Используется для задач классификации (например, предсказания следующего слова).
 *
 * @param predictions_logits Тензор сырых выходов модели (логитов) перед Softmax: [batchSize, numClasses].
 * @param targets_one_hot Тензор истинных меток в one-hot представлении: [batchSize, numClasses].
 * @returns Скалярный тензор, представляющий среднюю потерю по батчу.
 */
export function crossEntropyLossWithSoftmaxGrad(predictions_logits: Tensor, targets_one_hot: Tensor): Tensor {
    if (!predictions_logits.shape.every((dim, i) => dim === targets_one_hot.shape[i])) {
        throw new Error("Predictions logits and targets (one-hot) must have the same shape for Cross-Entropy loss.");
    }
    const [batchSize, numClasses] = predictions_logits.shape;

    // Сначала применяем Softmax к логитам, чтобы получить вероятности
    // Внимание: этот softmax не добавляется в граф для обратного прохода
    // так как его градиент будет обработан в 'gradFn' ниже.
    const softmaxProbs = softmax(predictions_logits); // Вычисляем softmax "вручную" для forward pass

    let totalLoss = 0;
    const epsilon = 1e-12; // Для численной стабильности, чтобы избежать log(0)

    // Вычисляем кросс-энтропию
    for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < numClasses; j++) {
            const predProb = softmaxProbs.data[i * numClasses + j];
            const targetVal = targets_one_hot.data[i * numClasses + j];

            if (targetVal === 1) { // Если это истинный класс (в one-hot представлении)
                totalLoss -= targetVal * Math.log(predProb + epsilon);
            }
        }
    }
    // Создаем тензор для средней потери
    const loss = new Tensor([totalLoss / batchSize], [1]);

    // Градиент для Softmax + Cross-Entropy комбинации: (вероятности - one-hot таргеты)
    // Этот градиент dLoss/d(logits) привязывается к predictions_logits
    loss._parents.push({
        tensor: predictions_logits,
        gradFn: (grad) => {
            const d_logits = new Float32Array(predictions_logits.size);
            for (let i = 0; i < predictions_logits.size; i++) {
                // `grad.data[0]` здесь - это градиент от операции `mean()` или просто 1.0,
                // если loss - это единственный тензор, от которого вызывается backward().
                d_logits[i] = (softmaxProbs.data[i] - targets_one_hot.data[i]) * grad.data[0];
            }
            return new Tensor(d_logits, predictions_logits.shape);
        }
    });

    return loss; // Возвращаем скалярный тензор со средней потерей
}

/**
 * Функция активации Softmax.
 * Преобразует вектор логитов в распределение вероятностей.
 * Используется отдельно для получения предсказаний после обучения.
 * @param input Входной тензор (логиты): [batchSize, numClasses].
 * @returns Тензор вероятностей: [batchSize, numClasses].
 */
export function softmax(input: Tensor): Tensor {
  if (input.shape.length !== 2) {
    throw new Error("Softmax expects 2D tensor [batch_size, num_classes].");
  }
  const [batchSize, numClasses] = input.shape;
  const resultData = new Float32Array(input.size);

  for (let i = 0; i < batchSize; i++) {
    const rowStart = i * numClasses;
    // Находим максимум для численной стабильности (вычитаем из каждого элемента)
    let maxVal = input.data[rowStart];
    for (let j = 1; j < numClasses; j++) {
      if (input.data[rowStart + j] > maxVal) {
        maxVal = input.data[rowStart + j];
      }
    }

    let sumExp = 0;
    for (let j = 0; j < numClasses; j++) {
      sumExp += Math.exp(input.data[rowStart + j] - maxVal);
    }

    for (let j = 0; j < numClasses; j++) {
      resultData[rowStart + j] = Math.exp(input.data[rowStart + j] - maxVal) / sumExp;
    }
  }
  // Softmax сам по себе не добавляет _parents, так как его градиент
  // обрабатывается в crossEntropyLossWithSoftmaxGrad для численной стабильности.
  return new Tensor(resultData, input.shape);
}
