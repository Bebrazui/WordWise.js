// src/utils/generation-algorithms.ts
import { Tensor } from '../lib/tensor';

/**
 * Применяет набор эвристик и алгоритмов для выбора следующего слова из логитов модели.
 * Это основной "мозг" генерации, который делает текст осмысленным.
 * 
 * @param predictionLogits - Сырые логиты от модели.
 * @param indexToWord - Маппинг для преобразования индекса в слово.
 * @param temperature - Контролирует случайность.
 * @param generatedSequence - Уже сгенерированная последовательность для контекста.
 * @returns - Объект с выбранным словом и топ-5 предсказаниями.
 */
export function sampleNextWord(
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
        
        for (let i = 0; i < vocabSize; i++) {
            const nextWord = indexToWord.get(i);
            if (!nextWord) continue;
            
            const potentialNgram = [...lastNMinus1Words, nextWord].join(' ');
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
