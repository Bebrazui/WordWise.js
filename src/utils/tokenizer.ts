
// src/utils/tokenizer.ts
import { Tensor } from '../lib/tensor';
import { softmax } from '../lib/layers';

/**
 * Строит словарь из текстового корпуса.
 * @param text Входной текстовый корпус.
 * @returns Объект со словарем (массив слов), маппингами слов в индексы и обратно, и размером словаря.
 */
export function buildTextVocabulary(text: string): { vocab: string[]; wordToIndex: Map<string, number>; indexToWord: Map<number, string>; vocabSize: number } {
  // Улучшенная токенизация: разбиваем на слова (латиница и кириллица) или <eos> токен, переводим в нижний регистр
  const words = text.toLowerCase().match(/<eos>|[a-zA-Zа-яА-ЯёЁ]+/g) || [];
  const uniqueWords = Array.from(new Set(words));
  const vocab = ['<unk>', ...uniqueWords]; // Добавляем токен для неизвестных слов на 0-й позиции
  const wordToIndex = new Map(vocab.map((word, i) => [word, i]));
  const indexToWord = new Map(vocab.map((word, i) => [i, word]));
  return { vocab, wordToIndex, indexToWord, vocabSize: vocab.length };
}

/**
 * Преобразует текстовый корпус в массив тензоров-последовательностей.
 * @param text Корпус.
 * @param wordToIndex Маппинг слов в индексы.
 * @param seqLen Длина последовательности.
 * @returns Массив тензоров-последовательностей.
 */
export function wordsToInputTensors(text: string, wordToIndex: Map<string, number>, seqLen: number): Tensor[] {
  const words = text.toLowerCase().match(/<eos>|[a-zA-Zа-яА-ЯёЁ]+/g) || [];
  const sequences: Tensor[] = [];
  if (words.length <= seqLen) return [];

  for (let i = 0; i < words.length - seqLen; i++) {
    const sequenceWords = words.slice(i, i + seqLen);
    const indices = sequenceWords.map(word => wordToIndex.get(word) || 0);
    sequences.push(new Tensor(indices, [seqLen]));
  }
  return sequences;
}


/**
 * Преобразует текстовый корпус в массив one-hot тензоров для целевых слов.
 * @param text Корпус.
 * @param wordToIndex Маппинг слов в индексы.
 * @param vocabSize Размер словаря.
 * @param seqLen Длина последовательности.
 * @returns Массив целевых тензоров.
 */
export function wordsToTargetTensors(text: string, wordToIndex: Map<string, number>, vocabSize: number, seqLen: number): Tensor[] {
  const words = text.toLowerCase().match(/<eos>|[a-zA-Zа-яА-ЯёЁ]+/g) || [];
  const targets: Tensor[] = [];
  if (words.length <= seqLen) return [];
  
  for (let i = 0; i < words.length - seqLen; i++) {
    const targetWord = words[i + seqLen];
    const targetIndex = wordToIndex.get(targetWord) || 0;
    const data = new Float32Array(vocabSize).fill(0);
    data[targetIndex] = 1;
    targets.push(new Tensor(data, [vocabSize]));
  }
  return targets;
}


/**
 * Выбирает слово из выходного тензора модели.
 * @param predictionLogits Тензор сырых логитов предсказания: [1, vocabSize].
 * @param indexToWord Карта индексов в слова.
 * @param temperature Параметр температуры для сэмплирования.
 * @returns Объект с выбранным словом и топ-5 предсказаниями.
 */
export function getWordFromPrediction(
  predictionLogits: Tensor,
  indexToWord: Map<number, string>,
  temperature: number = 1.0,
  generatedSequence: string[] = [],
): { chosenWord: string; topPredictions: { word: string; probability: number }[] } {
  if (predictionLogits.shape.length !== 2 || predictionLogits.shape[0] !== 1) {
    throw new Error(`Prediction tensor must have shape [1, vocabSize]. Got [${predictionLogits.shape}]`);
  }
  
  // Apply temperature to logits
  const logits = predictionLogits.divScalar(temperature);
  
  // Get probabilities using softmax
  const probabilitiesTensor = softmax(logits);
  const probabilities = Array.from(probabilitiesTensor.data);

  // --- Simple Repetition Penalty ---
  const penalty = 1.2;
  const recentWords = new Set(generatedSequence.slice(-5));
  for(const [idx, word] of indexToWord.entries()) {
      if (recentWords.has(word)) {
          probabilities[idx] /= penalty;
      }
  }

  // --- Get Top 5 Predictions for display ---
  const predictionsWithIndices = probabilities.map((probability, index) => ({
    word: indexToWord.get(index) || '<unk>',
    probability,
    index,
  }));
  predictionsWithIndices.sort((a, b) => b.probability - a.probability);
  const topPredictions = predictionsWithIndices.slice(0, 5);


  // --- Simple sampling from the adjusted probabilities ---
  const rand = Math.random();
  let cumulativeProb = 0;
  let chosenIndex = predictionsWithIndices[0].index; // Fallback to the best one
  
  // Renormalize probabilities after penalty
  const totalProb = predictionsWithIndices.reduce((sum, p) => sum + p.probability, 0);

  for (const pred of predictionsWithIndices) {
    cumulativeProb += pred.probability / totalProb;
    if (rand < cumulativeProb) {
      chosenIndex = pred.index;
      break;
    }
  }

  const chosenWord = indexToWord.get(chosenIndex) || '<unk>';

  return { chosenWord, topPredictions };
}
