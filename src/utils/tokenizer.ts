// src/utils/tokenizer.ts
import { Tensor } from '../lib/tensor';

/**
 * Строит словарь из текстового корпуса.
 * @param text Входной текстовый корпус.
 * @returns Объект со словарем (массив слов), маппингами слов в индексы и обратно, и размером словаря.
 */
export function buildVocabulary(text: string): { vocab: string[]; wordToIndex: Map<string, number>; indexToWord: Map<number, string>; vocabSize: number } {
  // Простая токенизация: разбиваем на слова, убираем знаки препинания, переводим в нижний регистр
  const words = text.toLowerCase().match(/\b\w+\b/g) || [];
  const uniqueWords = Array.from(new Set(words));
  const vocab = ['<unk>', ...uniqueWords]; // Добавляем токен для неизвестных слов на 0-й позиции
  const wordToIndex = new Map(vocab.map((word, i) => [word, i]));
  const indexToWord = new Map(vocab.map((word, i) => [i, word]));
  return { vocab, wordToIndex, indexToWord, vocabSize: vocab.length };
}

/**
 * Преобразует массив слов в массив тензоров, содержащих индексы этих слов.
 * Используется для входных данных модели.
 * @param words Массив слов.
 * @param wordToIndex Карта слов в индексы.
 * @returns Массив тензоров, каждый из которых содержит один индекс слова: [1].
 */
export function wordsToInputTensors(words: string[], wordToIndex: Map<string, number>): Tensor[] {
  return words.map(word => {
    const index = wordToIndex.get(word.toLowerCase()) || wordToIndex.get('<unk>')!;
    return new Tensor([index], [1]); // Тензор с одним элементом - индексом слова
  });
}

/**
 * Преобразует массив слов в массив тензоров в one-hot представлении.
 * Используется для целевых (истинных) меток.
 * @param words Массив слов.
 * @param wordToIndex Карта слов в индексы.
 * @param vocabSize Размер словаря.
 * @returns Массив тензоров, каждый из которых является one-hot вектором: [1, vocabSize].
 */
export function wordsToTargetTensors(words: string[], wordToIndex: Map<string, number>, vocabSize: number): Tensor[] {
  return words.map(word => {
    const index = wordToIndex.get(word.toLowerCase()) || wordToIndex.get('<unk>')!;
    const data = new Float32Array(vocabSize).fill(0);
    data[index] = 1;
    return new Tensor(data, [1, vocabSize]);
  });
}

/**
 * Преобразует числовой индекс в one-hot вектор.
 * @param index Индекс слова.
 * @param vocabSize Размер словаря.
 * @returns Тензор one-hot вектора: [1, vocabSize].
 */
export function indexToOneHot(index: number, vocabSize: number): Tensor {
  const data = new Float32Array(vocabSize).fill(0);
  data[index] = 1;
  return new Tensor(data, [1, vocabSize]);
}

/**
 * Выбирает слово с наибольшей вероятностью из выходного тензора модели.
 * @param predictionTensor Тензор вероятностей предсказания (после Softmax): [1, vocabSize].
 * @param indexToWord Карта индексов в слова.
 * @returns Предсказанное слово.
 */
export function getWordFromPrediction(predictionTensor: Tensor, indexToWord: Map<number, string>): string {
  if (predictionTensor.shape.length !== 2 || predictionTensor.shape[0] !== 1) {
    throw new Error("Prediction tensor for word selection must be [1, vocabSize].");
  }
  let maxProb = -1;
  let predictedIndex = -1;
  const vocabSize = predictionTensor.shape[1];

  for (let j = 0; j < vocabSize; j++) {
    if (predictionTensor.data[j] > maxProb) {
      maxProb = predictionTensor.data[j];
      predictedIndex = j;
    }
  }
  return indexToWord.get(predictedIndex) || '<unk>';
}

/**
 * Создает батчи из последовательности входных и целевых тензоров.
 * Для неполных батчей добавляет паддинг.
 * @param inputTensors Массив входных тензоров.
 * @param targetTensors Массив целевых тензоров.
 * @param batchSize Желаемый размер батча.
 * @returns Массив батчей, каждый из которых содержит объединенные входные и целевые тензоры.
 */
export function createBatches(inputTensors: Tensor[], targetTensors: Tensor[], batchSize: number): { inputs: Tensor, targets: Tensor }[] {
  const batches = [];
  const totalSteps = inputTensors.length;

  for (let i = 0; i < totalSteps; i += batchSize) {
    const currentInputBatch = inputTensors.slice(i, i + batchSize);
    const currentTargetBatch = targetTensors.slice(i, i + batchSize);

    const actualBatchSize = currentInputBatch.length;

    // Паддинг для последнего батча, если он неполный
    if (actualBatchSize < batchSize) {
        const paddingCount = batchSize - actualBatchSize;
        // Для входных индексов: добавляем индекс <unk> (0)
        const unkTensor = new Tensor([0], [1]);
        for (let k = 0; k < paddingCount; k++) {
            currentInputBatch.push(unkTensor);
        }
        // Для целевых one-hot векторов: добавляем нулевой вектор
        const zeroTarget = new Tensor(new Float32Array(targetTensors[0].size).fill(0), targetTensors[0].shape);
        for (let k = 0; k < paddingCount; k++) {
            currentTargetBatch.push(zeroTarget);
        }
    }
    
    // Объединяем тензоры в один батчевый тензор
    const batchedInputData = new Float32Array(batchSize);
    for(let j=0; j<batchSize; j++) {
        batchedInputData[j] = currentInputBatch[j].data[0];
    }
    
    const vocabSize = currentTargetBatch[0].shape[1];
    const batchedTargetData = new Float32Array(batchSize * vocabSize);
     for(let j=0; j<batchSize; j++) {
        batchedTargetData.set(currentTargetBatch[j].data, j * vocabSize);
    }
    
    const batchedInputTensor = new Tensor(batchedInputData, [batchSize, 1]); // [batchSize, 1]
    const batchedTargetTensor = new Tensor(batchedTargetData, [batchSize, vocabSize]); // [batchSize, vocabSize]

    batches.push({ inputs: batchedInputTensor, targets: batchedTargetTensor });
  }
  return batches;
}
