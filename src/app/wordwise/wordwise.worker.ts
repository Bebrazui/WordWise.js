
// src/app/wordwise/wordwise.worker.ts
/// <reference lib="webworker" />

import { WordWiseModel, serializeModel } from '@/lib/model';
import { SGD } from '@/lib/optimizer';
import { crossEntropyLossWithSoftmaxGrad } from '@/lib/layers';
import { buildVocabulary, wordsToInputTensors, wordsToTargetTensors, createBatches, indexToOneHot } from '@/utils/tokenizer';
import { Tensor } from '@/lib/tensor';

let model: WordWiseModel | null = null;
let optimizer: SGD | null = null;
let vocabData: { vocab: string[]; wordToIndex: Map<string, number>; indexToWord: Map<number, string>; vocabSize: number } | null = null;
let trainingStopFlag = false;

/**
 * Обработчик сообщений от основного потока.
 * Управляет инициализацией, обучением и остановкой модели.
 */
self.onmessage = async (event: MessageEvent) => {
  const { type, payload } = event.data;

  try {
    switch (type) {
      case 'initialize':
        initialize(payload);
        break;
      case 'train':
        trainingStopFlag = false;
        await train(payload);
        break;
      case 'stop':
        trainingStopFlag = true;
        break;
      case 'generate':
        generate(payload);
        break;
    }
  } catch (error) {
    self.postMessage({ type: 'error', payload: { message: error instanceof Error ? error.message : String(error) } });
  }
};

/**
 * Инициализирует модель, словарь и оптимизатор.
 */
function initialize(payload: { textCorpus: string; embeddingDim: number; hiddenSize: number; learningRate: number }) {
  const { textCorpus, embeddingDim, hiddenSize, learningRate } = payload;
  
  // Используем регулярное выражение для токенизации, чтобы включить кириллицу
  const words = textCorpus.toLowerCase().match(/[a-zа-яё]+/g) || [];
  vocabData = buildVocabulary(words.join(' '));
  
  model = new WordWiseModel(vocabData.vocabSize, embeddingDim, hiddenSize);
  optimizer = new SGD(learningRate);

  const wordsForSampling = vocabData.vocab.filter(w => !['<unk>', 'вопрос', 'ответ'].includes(w) && w.length > 2);
  const shuffled = wordsForSampling.sort(() => 0.5 - Math.random());
  
  self.postMessage({ 
    type: 'initialized', 
    payload: { 
      vocabSize: vocabData.vocabSize,
      sampleWords: shuffled.slice(0, 4)
    } 
  });
}

/**
 * Асинхронно обучает модель.
 */
async function train(payload: { textCorpus: string, numEpochs: number, learningRate: number, batchSize: number, lossHistory: {epoch: number, loss: number}[] }) {
  if (!model || !vocabData || !optimizer) {
    throw new Error('Модель не инициализирована.');
  }

  const { textCorpus, numEpochs, learningRate, batchSize, lossHistory } = payload;
  optimizer.learningRate = learningRate;

  const words = textCorpus.toLowerCase().match(/[a-zа-яё]+/g) || [];
  if (words.length < 2) {
    throw new Error('Недостаточно слов для обучения в корпусе.');
  }

  const inputTensors = wordsToInputTensors(words.slice(0, -1), vocabData.wordToIndex);
  const targetTensors = wordsToTargetTensors(words.slice(1), vocabData.wordToIndex, vocabData.vocabSize);
  const batches = createBatches(inputTensors, targetTensors, batchSize);

  const startEpoch = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].epoch + 1 : 1;
  
  for (let epoch = 0; epoch < numEpochs; epoch++) {
    if (trainingStopFlag) {
      self.postMessage({ type: 'training-stopped', payload: { epoch: startEpoch + epoch } });
      return;
    }

    let epochLoss = 0;
    
    // Перебираем батчи для одной эпохи
    for (const batch of batches) {
      // Состояния нужно сбрасывать для каждого нового батча, так как последовательности в батчах независимы.
      // Но для stateful RNN между батчами одной эпохи состояние можно сохранять.
      // Для простоты здесь мы сбрасываем состояние для каждого батча.
      let h = model.initializeStates(batchSize).h0;
      let c = model.initializeStates(batchSize).c0;

      const { outputLogits: predictionLogits, h: nextH, c: nextC } = model.forwardStep(batch.inputs, h, c);
      h = nextH.detach(); // Отсоединяем от графа, чтобы градиент не тёк между батчами
      c = nextC.detach();

      const lossTensor = crossEntropyLossWithSoftmaxGrad(predictionLogits, batch.targets);
      epochLoss += lossTensor.data[0];
      
      // Обратный проход и шаг оптимизатора
      lossTensor.backward();
      optimizer.step(model.getParameters());
    }

    const avgEpochLoss = epochLoss / batches.length;
    const currentEpochNumber = startEpoch + epoch;

    self.postMessage({
      type: 'progress',
      payload: {
        epoch: currentEpochNumber,
        loss: avgEpochLoss,
        progress: ((epoch + 1) / numEpochs) * 100,
      },
    });

    // Даем главному потоку время на обработку сообщений
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  
  // Сериализуем обученную модель и отправляем обратно
  const modelJson = serializeModel(model, vocabData);
  self.postMessage({ type: 'training-complete', payload: { modelJson } });
}


/**
 * Генерирует текст на основе начального слова.
 */
function generate(payload: { startWord: string, numWords: number, temperature: number }) {
    if (!model || !vocabData) {
        throw new Error('Генерация невозможна: модель не обучена.');
    }
    
    const { startWord, numWords, temperature } = payload;
    const { wordToIndex, indexToWord } = vocabData;

    let currentWord = startWord.toLowerCase();
    let initialOutput = '';
    if (!wordToIndex.has(currentWord)) {
        currentWord = '<unk>';
        initialOutput = `Начальное слово "${startWord}" не найдено. Используем "<unk>".\n`;
    }

    let generatedSequence = [currentWord];
    let h = model.initializeStates(1).h0;
    let c = model.initializeStates(1).c0;
    
    for (let i = 0; i < numWords; i++) {
        const inputTensor = new Tensor([wordToIndex.get(currentWord) || 0], [1]);
        const { outputLogits, h: nextH, c: nextC } = model.forwardStep(inputTensor, h, c);
        h = nextH;
        c = nextC;

        // Note: For full functionality, getWordFromPrediction logic should be here
        // or the function needs to be available in the worker scope.
        // Simplified greedy search for now:
        let maxIdx = 0;
        let maxVal = -Infinity;
        for (let j = 0; j < outputLogits.size; j++) {
            if (outputLogits.data[j] > maxVal) {
                maxVal = outputLogits.data[j];
                maxIdx = j;
            }
        }
        const chosenWord = indexToWord.get(maxIdx) || '<unk>';
        
        if (chosenWord === 'вопрос' || chosenWord === 'ответ') {
            continue;
        }

        generatedSequence.push(chosenWord);
        currentWord = chosenWord;

        if (chosenWord === '<unk>') {
            break;
        }
    }

    const generatedText = initialOutput + `Сгенерированный текст: ${generatedSequence.join(' ')}`;
    self.postMessage({ type: 'generation-result', payload: { text: generatedText } });
}

self.postMessage({ type: 'worker-ready' });

    