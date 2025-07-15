
// src/utils/tokenizer.ts
import { Tensor } from '../lib/tensor';
import { sampleNextWord } from './generation-algorithms';

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
  // For sequence models, the target for word at index `i` is the word at index `i+1`.
  // Here, we'll just one-hot encode the words themselves, and the batching function will handle the sequence logic.
  return words.map(word => {
    const index = wordToIndex.get(word.toLowerCase()) || wordToIndex.get('<unk>')!;
    const data = new Float32Array(vocabSize).fill(0);
    data[index] = 1;
    return new Tensor(data, [1, vocabSize]);
  });
}


/**
 * Выбирает слово из выходного тензора модели.
 * Эта функция является оберткой над сложной логикой в `generation-algorithms.ts`.
 * @param predictionLogits Тензор сырых логитов предсказания: [1, vocabSize].
 * @param indexToWord Карта индексов в слова.
 * @param temperature Параметр температуры для сэмплирования.
 * @param generatedSequence Массив уже сгенерированных слов для применения штрафа.
 * @returns Объект с выбранным словом и топ-5 предсказаниями.
 */
export function getWordFromPrediction(
  predictionLogits: Tensor,
  indexToWord: Map<number, string>,
  temperature: number = 1.0,
  generatedSequence: string[] = [],
): { chosenWord: string; topPredictions: { word: string; probability: number }[] } {
    return sampleNextWord(
        predictionLogits,
        indexToWord,
        temperature,
        generatedSequence
    );
}
