// src/utils/tokenizer.ts
import { Tensor } from '../lib/tensor';

/**
 * Строит словарь из текстового корпуса.
 * @param text Входной текстовый корпус.
 * @returns Объект со словарем (массив слов), маппингами слов в индексы и обратно, и размером словаря.
 */
export function buildTextVocabulary(text: string): { vocab: string[]; wordToIndex: Map<string, number>; indexToWord: Map<number, string>; vocabSize: number } {
  // Улучшенная токенизация: разбиваем на слова (латиница и кириллица), переводим в нижний регистр
  const words = text.toLowerCase().match(/[a-zA-Zа-яА-ЯёЁ]+/g) || [];
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
 * Выбирает слово из выходного тензора модели с учетом температуры и штрафа за повторение.
 * @param predictionLogits Тензор сырых логитов предсказания: [1, vocabSize].
 * @param indexToWord Карта индексов в слова.
 * @param temperature Параметр температуры для сэмплирования.
 * @param generatedSequence Массив уже сгенерированных слов для применения штрафа.
 * @param repetitionPenalty Сила штрафа за повторение (например, 1.2).
 * @returns Объект с выбранным словом и топ-5 предсказаниями.
 */
export function getWordFromPrediction(
  predictionLogits: Tensor,
  indexToWord: Map<number, string>,
  temperature: number = 1.0,
  generatedSequence: string[] = [],
  repetitionPenalty: number = 1.2
): { chosenWord: string; topPredictions: { word: string; probability: number }[] } {
  if (predictionLogits.shape.length !== 2 || predictionLogits.shape[0] !== 1) {
    throw new Error("Prediction tensor for word selection must be [1, vocabSize].");
  }
  const vocabSize = predictionLogits.shape[1];
  const logits = predictionLogits.data.slice(); // Копируем, чтобы не изменять исходные логиты

  // --- 1. Применяем штраф за повторение ---
  const generatedWordSet = new Set(generatedSequence);
  for (let i = 0; i < vocabSize; i++) {
    const word = indexToWord.get(i);
    if (word && generatedWordSet.has(word)) {
      // Понижаем логиты для уже сгенерированных слов
      // Если логит положительный, делим на штраф; если отрицательный, умножаем
      logits[i] = logits[i] > 0 ? logits[i] / repetitionPenalty : logits[i] * repetitionPenalty;
    }
  }

  // --- 2. Применяем температуру и Softmax ---
  const adjustedProbs = new Float32Array(vocabSize);
  let sumExp = 0;

  // Находим максимум для численной стабильности
  let maxLogit = -Infinity;
  for (let j = 0; j < vocabSize; j++) {
    if (logits[j] > maxLogit) {
      maxLogit = logits[j];
    }
  }

  for (let j = 0; j < vocabSize; j++) {
    const exponentiated = Math.exp((logits[j] - maxLogit) / temperature);
    adjustedProbs[j] = exponentiated;
    sumExp += exponentiated;
  }

  // Нормализуем для получения вероятностей
  for (let j = 0; j < vocabSize; j++) {
    adjustedProbs[j] /= sumExp;
  }

  // --- 3. Находим топ-5 предсказаний для визуализации (уже с учетом штрафов и температуры) ---
  const allPredictions = Array.from(adjustedProbs).map((probability, index) => ({
    word: indexToWord.get(index) || '<unk>',
    probability,
    index,
  }));

  allPredictions.sort((a, b) => b.probability - a.probability);
  const topPredictions = allPredictions.slice(0, 5);

  // --- 4. Выполняем выборку (сэмплирование) ---
  let randomValue = Math.random();
  let cumulativeProbability = 0;
  let predictedIndex = -1;

  for (let j = 0; j < vocabSize; j++) {
    cumulativeProbability += adjustedProbs[j];
    if (randomValue <= cumulativeProbability) {
      predictedIndex = j;
      break;
    }
  }

  if (predictedIndex === -1) {
    predictedIndex = 0; // Fallback to <unk>
  }

  const chosenWord = indexToWord.get(predictedIndex) || '<unk>';

  return { chosenWord, topPredictions };
}
