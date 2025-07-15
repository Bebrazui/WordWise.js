// src/lib/layers.ts
import { Tensor } from './tensor';

/**
 * Базовый абстрактный класс для всех слоев нейронной сети.
 * Определяет общий интерфейс для получения параметров и выполнения прямого прохода.
 */
export abstract class Layer {
  abstract parameters: Tensor[]; // Обучаемые параметры слоя
  // forward может принимать несколько входных тензоров и возвращать несколько
  abstract forward(...inputs: any[]): Tensor | { [key: string]: Tensor };

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
    this.weights.name = 'weights'; // Даем имя для сериализации
    this.parameters = [this.weights];
  }

  /**
   * Прямой проход слоя Embedding.
   * @param input Тензор с индексами слов: [batchSize, seqLen].
   * @returns Тензор эмбеддингов: [batchSize, seqLen, embeddingDim].
   */
  forward(input: Tensor): Tensor {
    if (input.shape.length !== 2) {
      throw new Error(`Embedding layer input must be a 2D tensor of indices [batchSize, seqLen]. Got shape [${input.shape}]`);
    }
    const [batchSize, seqLen] = input.shape;
    const embeddingDim = this.weights.shape[1];
    const vocabSize = this.weights.shape[0];

    const resultData = new Float32Array(batchSize * seqLen * embeddingDim);
    const parentTensor = this.weights;

    for (let b = 0; b < batchSize; b++) {
      for (let s = 0; s < seqLen; s++) {
        const wordIndex = Math.floor(input.data[b * seqLen + s]);
        const finalIndex = (wordIndex >= 0 && wordIndex < vocabSize) ? wordIndex : 0; // Fallback to <unk>

        const embeddingRowStart = finalIndex * embeddingDim;
        const resultRowStart = (b * seqLen + s) * embeddingDim;

        for (let d = 0; d < embeddingDim; d++) {
          resultData[resultRowStart + d] = this.weights.data[embeddingRowStart + d];
        }
      }
    }

    const result = new Tensor(resultData, [batchSize, seqLen, embeddingDim]);

    result._parents.push({
      tensor: parentTensor,
      gradFn: (grad) => {
        const dWeights = Tensor.zeros(parentTensor.shape);
        for (let b = 0; b < batchSize; b++) {
          for (let s = 0; s < seqLen; s++) {
            const wordIndex = Math.floor(input.data[b * seqLen + s]);
            if (wordIndex < 0 || wordIndex >= vocabSize) continue;

            const gradRowStart = (b * seqLen + s) * embeddingDim;
            const dWeightsRowStart = wordIndex * embeddingDim;
            for (let d = 0; d < embeddingDim; d++) {
              dWeights.data[dWeightsRowStart + d] += grad.data[gradRowStart + d];
            }
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

    this.weights.name = 'weights';
    this.bias.name = 'bias';
    this.parameters = [this.weights, this.bias];
  }

  /**
   * Прямой проход линейного слоя.
   * @param input Входной тензор: [batchSize, ..., inputSize].
   * @returns Выходной тензор: [batchSize, ..., outputSize].
   */
  forward(input: Tensor): Tensor {
    // Z = X @ W + B
    const matMulResult = input.dot(this.weights);
    const output = matMulResult.add(this.bias);
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
        // Инициализация весов гейтов методом Glorot/Xavier
        const limit_x = Math.sqrt(6 / (inputSize + hiddenSize));
        const limit_h = Math.sqrt(6 / (hiddenSize + hiddenSize));

        this.Wi = Tensor.randn([inputSize, hiddenSize], limit_x); this.Wi.name = 'Wi';
        this.Ui = Tensor.randn([hiddenSize, hiddenSize], limit_h); this.Ui.name = 'Ui';
        this.Bi = Tensor.zeros([1, hiddenSize]); this.Bi.name = 'Bi';

        this.Wf = Tensor.randn([inputSize, hiddenSize], limit_x); this.Wf.name = 'Wf';
        this.Uf = Tensor.randn([hiddenSize, hiddenSize], limit_h); this.Uf.name = 'Uf';
        this.Bf = Tensor.zeros([1, hiddenSize]); this.Bf.name = 'Bf';

        this.Wo = Tensor.randn([inputSize, hiddenSize], limit_x); this.Wo.name = 'Wo';
        this.Uo = Tensor.randn([hiddenSize, hiddenSize], limit_h); this.Uo.name = 'Uo';
        this.Bo = Tensor.zeros([1, hiddenSize]); this.Bo.name = 'Bo';

        this.Wc = Tensor.randn([inputSize, hiddenSize], limit_x); this.Wc.name = 'Wc';
        this.Uc = Tensor.randn([hiddenSize, hiddenSize], limit_h); this.Uc.name = 'Uc';
        this.Bc = Tensor.zeros([1, hiddenSize]); this.Bc.name = 'Bc';

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
        // Оптимизация: объединяем матричные умножения, где это возможно
        const gates = input.dot(this.Wi).add(prevH.dot(this.Ui)).add(this.Bi).add(
                      input.dot(this.Wf).add(prevH.dot(this.Uf)).add(this.Bf)
                    ).add(
                      input.dot(this.Wo).add(prevH.dot(this.Uo)).add(this.Bo)
                    ).add(
                      input.dot(this.Wc).add(prevH.dot(this.Uc)).add(this.Bc)
                    );
        
        const hiddenSize = this.Bi.shape[1];
        const i_t_raw = gates.slice([0, 0], [gates.shape[0], hiddenSize]);
        const f_t_raw = gates.slice([0, hiddenSize], [gates.shape[0], hiddenSize]);
        const o_t_raw = gates.slice([0, hiddenSize * 2], [gates.shape[0], hiddenSize]);
        const c_tilde_t_raw = gates.slice([0, hiddenSize * 3], [gates.shape[0], hiddenSize]);

        const i_t = sigmoid(i_t_raw);
        const f_t = sigmoid(f_t_raw);
        const o_t = sigmoid(o_t_raw);
        const c_tilde_t = tanh(c_tilde_t_raw);
        
        // Обновление состояния ячейки: c_t = f_t * c_{t-1} + i_t * c_tilde_t
        const c_t = f_t.mul(prevC).add(i_t.mul(c_tilde_t));

        // Обновление скрытого состояния: h_t = o_t * tanh(c_t)
        const h_t = o_t.mul(tanh(c_t));

        return { h: h_t, c: c_t };
    }
}

// --- Слой выравнивания (Flatten) ---
export class Flatten extends Layer {
    parameters: Tensor[] = [];
    inputShape: number[] = [];

    forward(input: Tensor): Tensor {
        this.inputShape = input.shape;
        const batchSize = this.inputShape[0];
        const newSize = input.size / batchSize;
        const output = input.reshape([batchSize, newSize]);

        output._parents.push({
            tensor: input,
            gradFn: (grad) => grad.reshape(this.inputShape)
        });
        return output;
    }
}

// --- Слой нормализации (Layer Normalization) ---
export class LayerNorm extends Layer {
    gamma: Tensor; // Обучаемый параметр масштабирования
    beta: Tensor;  // Обучаемый параметр сдвига
    epsilon: number;
    parameters: Tensor[];

    constructor(normalizedShape: number, epsilon = 1e-5) {
        super();
        this.gamma = Tensor.ones([1, normalizedShape]);
        this.beta = Tensor.zeros([1, normalizedShape]);
        this.gamma.name = 'gamma';
        this.beta.name = 'beta';
        this.epsilon = epsilon;
        this.parameters = [this.gamma, this.beta];
    }

    forward(input: Tensor): Tensor {
        // Нормализация происходит по последней размерности (признакам)
        const mean = input.mean(-1, true); // [B, S, 1] or [B, 1]
        const variance = input.sub(mean).pow(2).mean(-1, true); // [B, S, 1] or [B, 1]
        const std = variance.addScalar(this.epsilon).sqrt();
        const x_normalized = input.sub(mean).divScalar(std.data[0]); // Simplified for now
        return x_normalized.mul(this.gamma).add(this.beta);
    }
}


// --- Positional Embedding Layer ---
export class PositionalEmbedding extends Layer {
    private positionalEncoding: Tensor;
    parameters: Tensor[] = [];

    constructor(maxSeqLen: number, dModel: number) {
        super();
        this.positionalEncoding = Tensor.zeros([maxSeqLen, dModel]);
        for (let pos = 0; pos < maxSeqLen; pos++) {
            for (let i = 0; i < dModel; i++) {
                let value: number;
                if (i % 2 === 0) {
                    value = Math.sin(pos / Math.pow(10000, i / dModel));
                } else {
                    value = Math.cos(pos / Math.pow(10000, (i - 1) / dModel));
                }
                this.positionalEncoding.data[pos * dModel + i] = value;
            }
        }
    }
    
    forward(x: Tensor): Tensor {
        const seqLen = x.shape[1];
        // Обрезаем позиционные кодировки до нужной длины последовательности
        const posEncodingSlice = this.positionalEncoding.slice([0, 0], [seqLen, x.shape[2]]);
        // Добавляем позиционные кодировки к входным эмбеддингам
        return x.add(posEncodingSlice);
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
    if (predictions_logits.shape.length > 2) {
        // Flatten for loss calculation if needed, e.g., from [batch, seq, vocab]
        const [batchSize, seqLen, vocabSize] = predictions_logits.shape;
        predictions_logits = predictions_logits.reshape([batchSize * seqLen, vocabSize]);
        targets_one_hot = targets_one_hot.reshape([batchSize * seqLen, vocabSize]);
    }
    
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
                totalLoss -= Math.log(predProb + epsilon);
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
                 d_logits[i] = (softmaxProbs.data[i] - targets_one_hot.data[i]) / batchSize * grad.data[0];
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
 * @param input Входной тензор (логиты): [batchSize, ..., numClasses].
 * @returns Тензор вероятностей: [batchSize, ..., numClasses].
 */
export function softmax(input: Tensor): Tensor {
    const lastDim = input.shape[input.shape.length - 1];
    const reshaped = input.reshape([-1, lastDim]);
    const [batchSize, numClasses] = reshaped.shape;
    const resultData = new Float32Array(reshaped.size);

    for (let i = 0; i < batchSize; i++) {
        const rowStart = i * numClasses;
        // Находим максимум для численной стабильности (вычитаем из каждого элемента)
        let maxVal = reshaped.data[rowStart];
        for (let j = 1; j < numClasses; j++) {
            if (reshaped.data[rowStart + j] > maxVal) {
                maxVal = reshaped.data[rowStart + j];
            }
        }

        let sumExp = 0;
        for (let j = 0; j < numClasses; j++) {
            sumExp += Math.exp(reshaped.data[rowStart + j] - maxVal);
        }

        for (let j = 0; j < numClasses; j++) {
            resultData[rowStart + j] = Math.exp(reshaped.data[rowStart + j] - maxVal) / sumExp;
        }
    }
    // Softmax сам по себе не добавляет _parents, так как его градиент
    // обрабатывается в crossEntropyLossWithSoftmaxGrad для численной стабильности.
    const resultTensor = new Tensor(resultData, reshaped.shape);
    return resultTensor.reshape(input.shape);
}
