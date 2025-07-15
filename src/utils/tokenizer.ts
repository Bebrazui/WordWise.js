
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
 * Выбирает слово из выходного тензора модели с учетом температуры и штрафа за повторение.
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
    if (predictionLogits.shape.length !== 2 || predictionLogits.shape[0] !== 1) {
        throw new Error(`Prediction tensor for word selection must be [1, vocabSize]. Got [${predictionLogits.shape}]`);
    }
    const vocabSize = predictionLogits.shape[1];
    const logits = predictionLogits.data.slice(); // Копируем, чтобы не изменять исходные логиты

    // --- АЛГОРИТМ 1: Штраф за повторение (Repetition Penalty) ---
    // Уменьшаем вероятность слов, которые недавно появлялись.
    const REPETITION_PENALTY = 1.8;
    const penaltyWindow = Math.min(generatedSequence.length, 10);
    const recentWords = new Set(generatedSequence.slice(-penaltyWindow));
    
    for (const word of recentWords) {
        // Находим индекс слова и применяем штраф к его логиту
        for (const [key, value] of indexToWord.entries()) {
            if (value === word) {
                const idx = key;
                logits[idx] = logits[idx] < 0 ? logits[idx] * REPETITION_PENALTY : logits[idx] / REPETITION_PENALTY;
                break;
            }
        }
    }
    
    // --- АЛГОРИТМ 2: Блокировка N-граммов ---
    // Запрещаем генерацию n-граммов (последовательностей), которые уже были в тексте.
    const NGRAM_SIZE = 2; // Блокируем повторение пар слов (биграммов)
    if (generatedSequence.length >= NGRAM_SIZE -1) {
        const lastNMinus1Words = generatedSequence.slice(-(NGRAM_SIZE - 1));
        const ngramToBlock = [...lastNMinus1Words, ''].join(' '); // формируем начало биграмма

        for (let i = 0; i < vocabSize; i++) {
            const nextWord = indexToWord.get(i);
            if (!nextWord) continue;
            
            const potentialNgram = ngramToBlock + nextWord;
            if (generatedSequence.join(' ').includes(potentialNgram)) {
                logits[i] = -Infinity; // Блокируем этот токен
            }
        }
    }


    // --- АЛГОРИТМ 3: Подавление специальных и коротких токенов ---
    for (let i = 0; i < vocabSize; i++) {
        const word = indexToWord.get(i);
        if (word === '<unk>' || word === 'вопрос' || word === 'ответ' || (word && word.length < 2)) {
            logits[i] = -Infinity; // Полностью блокируем эти токены
        }
    }

    // --- Применяем температуру ---
    for (let i = 0; i < vocabSize; i++) {
        logits[i] /= temperature;
    }

    // --- Softmax для получения вероятностей ---
    let maxLogit = -Infinity;
    for (const l of logits) {
        if (isFinite(l) && l > maxLogit) {
            maxLogit = l;
        }
    }

    const probabilities = new Float32Array(vocabSize);
    let sumExp = 0;
    for (let i = 0; i < vocabSize; i++) {
        if (isFinite(logits[i])) {
            const exp = Math.exp(logits[i] - maxLogit);
            probabilities[i] = exp;
            sumExp += exp;
        } else {
            probabilities[i] = 0;
        }
    }
    
    if (sumExp === 0) { // Если все заблокированы, выбираем любой не-inf токен
        const availableIndex = logits.findIndex(l => isFinite(l));
        const chosenWord = indexToWord.get(availableIndex) || '<unk>';
        return { chosenWord, topPredictions: [{word: chosenWord, probability: 1.0}]};
    }

    for (let i = 0; i < vocabSize; i++) {
        probabilities[i] /= sumExp;
    }

    // --- АЛГОРИТМ 4: Top-K Sampling ---
    // Выбираем слово не из всего словаря, а только из K наиболее вероятных.
    const K = 20; 
    const allPredictions = Array.from(probabilities)
        .map((probability, index) => ({
            word: indexToWord.get(index) || '<unk>',
            probability,
            index,
        }))
        .filter(p => p.probability > 0); // Отфильтровываем заблокированные

    allPredictions.sort((a, b) => b.probability - a.probability);
    
    const topKPredictions = allPredictions.slice(0, K);
    
    if (topKPredictions.length === 0) {
        const fallbackWord = indexToWord.get(0) || '<unk>';
        return { chosenWord: fallbackWord, topPredictions: [] };
    }
    
    // Пересчитываем вероятности внутри Top-K
    const topKSum = topKPredictions.reduce((sum, p) => sum + p.probability, 0);
    const topKNormalized = topKPredictions.map(p => ({ ...p, probability: p.probability / topKSum }));

    // --- Финальный выбор слова (сэмплирование) ---
    let randomValue = Math.random();
    let cumulativeProbability = 0;
    let predictedIndex = topKNormalized[0].index; // Fallback to the best one

    for (const pred of topKNormalized) {
        cumulativeProbability += pred.probability;
        if (randomValue <= cumulativeProbability) {
            predictedIndex = pred.index;
            break;
        }
    }

    const chosenWord = indexToWord.get(predictedIndex) || '<unk>';

    return { chosenWord, topPredictions: allPredictions.slice(0, 5) };
}
