
// src/app/wordwise/wordwise.worker.ts
/// <reference lib="webworker" />

import { WordWiseModel, TransformerModel, FlowNetModel, serializeModel, deserializeModel, AnyModel, VocabData, FitCallbacks } from '@/lib/model';
import { buildTextVocabulary, wordsToInputTensors, wordsToTargetTensors, getWordFromPrediction } from '@/utils/tokenizer';
import { Tensor } from '@/lib/tensor';

let model: AnyModel | null = null;
let vocabData: VocabData | null = null;
let trainingData: {
    inputs: Tensor[],
    targets: Tensor[],
} | null = null;
let modelType: 'lstm' | 'transformer' | 'flownet' = 'flownet';


/**
 * Handles messages from the main thread.
 */
self.onmessage = async (event: MessageEvent) => {
  const { type, payload } = event.data;

  try {
    switch (type) {
      case 'initialize':
        await initialize(payload);
        break;
      case 'train':
        await train(payload);
        break;
      case 'stop':
         if (model) {
            model.stopTraining = true;
         }
        break;
      case 'load-model':
        await loadModel(payload);
        break;
      case 'generate':
        await generate(payload);
        break;
    }
  } catch (error) {
    self.postMessage({ type: 'error', payload: { message: error instanceof Error ? error.message : String(error), error } });
  }
};


async function loadModel(payload: {modelJson: string, textCorpus: string}) {
    const { modelJson, textCorpus } = payload;
    const loaded = deserializeModel(modelJson);
    model = loaded.model;
    vocabData = loaded.vocabData;
    
    trainingData = null;

    if ('vocab' in vocabData && textCorpus) {
        const words = textCorpus.toLowerCase().match(/[a-zа-яё]+/g) || [];
        trainingData = {
           inputs: wordsToInputTensors(words, vocabData.wordToIndex),
           targets: wordsToTargetTensors(words, vocabData.wordToIndex, vocabData.vocabSize)
        };
    } else {
        throw new Error("Cannot load a model without a text corpus to create training data.");
    }


    if (model instanceof TransformerModel) {
        modelType = 'transformer';
    } else if (model instanceof WordWiseModel) {
        modelType = 'lstm';
    } else if (model instanceof FlowNetModel) {
        modelType = 'flownet';
    } else {
        throw new Error("Unknown model type after loading.");
    }

    const wordsForSampling = ('vocab' in vocabData && vocabData.vocab) ? vocabData.vocab.filter(w => !['<unk>', 'вопрос', 'ответ'].includes(w) && w.length > 2) : [];
    const shuffled = wordsForSampling.sort(() => 0.5 - Math.random());
    
    self.postMessage({
        type: 'model-loaded',
        payload: {
            architecture: (model as any).getArchitecture(),
            sampleWords: shuffled.slice(0, 4)
        }
    });
}


/**
 * Initializes the model, vocabulary, and training data.
 */
async function initialize(payload: any) {
  const { modelType: newModelType, textCorpus } = payload;
  modelType = newModelType;
  vocabData = buildTextVocabulary(textCorpus);
  
  if (modelType === 'lstm') {
    const { embeddingDim, hiddenSize } = payload;
    model = new WordWiseModel(vocabData.vocabSize, embeddingDim, hiddenSize);
  } else if (modelType === 'transformer') {
    const { dModel, numHeads, dff, numLayers, seqLen } = payload;
    model = new TransformerModel(vocabData.vocabSize, seqLen, dModel, numLayers, numHeads, dff);
  } else if (modelType === 'flownet') {
    const { embeddingDim, numLayers, seqLen } = payload;
    model = new FlowNetModel(vocabData.vocabSize, seqLen, embeddingDim, numLayers);
  } else {
    throw new Error('Unknown initialization type');
  }

  const words = textCorpus.toLowerCase().match(/[a-zа-яё]+/g) || [];
  trainingData = {
     inputs: wordsToInputTensors(words, vocabData.wordToIndex),
     targets: wordsToTargetTensors(words, vocabData.wordToIndex, vocabData.vocabSize)
  };
  
  const wordsForSampling = vocabData.vocab.filter(w => !['<unk>', 'вопрос', 'ответ'].includes(w) && w.length > 2);
  const shuffled = wordsForSampling.sort(() => 0.5 - Math.random());

  self.postMessage({ 
    type: 'initialized', 
    payload: { 
      type: modelType,
      vocabSize: vocabData.vocabSize,
      sampleWords: shuffled.slice(0, 4)
    } 
  });
}

/**
 * Trains the model using the model.fit() method.
 */
async function train(payload: { numEpochs: number, learningRate: number, batchSize: number, lossHistory: {epoch: number, loss: number}[] }) {
  if (!model || !vocabData || !trainingData) {
    self.postMessage({ type: 'error', payload: { message: 'Model is not initialized or no training data is available.' } });
    return;
  }

  const { numEpochs, learningRate, batchSize, lossHistory } = payload;
  model.stopTraining = false;
  
  const callbacks: FitCallbacks = {
    onEpochEnd: (log) => {
        self.postMessage({
            type: 'progress',
            payload: {
                epoch: log.epoch,
                loss: log.loss,
                gradients: log.gradients,
            },
        });
        return model?.stopTraining || false;
    }
  };

  await model.fit(trainingData.inputs, trainingData.targets, {
      epochs: numEpochs,
      batchSize,
      learningRate,
      initialEpoch: lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].epoch : 0
  }, callbacks);
  
  if (model.stopTraining && model && vocabData) {
       const lastEpoch = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].epoch : 0;
       const modelJson = serializeModel(model, vocabData);
       self.postMessage({ type: 'training-stopped', payload: { epoch: lastEpoch, modelJson } });
       return;
  }
  
  const modelJson = serializeModel(model, vocabData);
  self.postMessage({ type: 'training-complete', payload: { modelJson } });
}


async function generate(payload: {startWord: string, numWords: number, temperature: number}) {
    if (!model || !vocabData || !('wordToIndex' in vocabData)) {
        throw new Error("Text model not ready for generation.");
    }
    const { startWord, numWords, temperature } = payload;
    const { wordToIndex, indexToWord } = vocabData;

    let currentWordSequence = startWord.toLowerCase().split(' ').filter(Boolean);

    for (let i = 0; i < numWords; i++) {
        let chosenWord: string;
        let topPredictions: any[] = [];

        if (model instanceof WordWiseModel) {
            // LSTM generation is stateful and one-word-at-a-time, not implemented here for streaming
            throw new Error("Streaming generation for LSTM not implemented in this worker.");
        } else if (model instanceof TransformerModel || model instanceof FlowNetModel) {
            const currentSequenceIndices = currentWordSequence.slice(-model.seqLen).map(w => wordToIndex.get(w) || 0);
            while(currentSequenceIndices.length < model.seqLen) {
                currentSequenceIndices.unshift(0); // Pad with <unk>
            }
            const inputTensor = new Tensor(currentSequenceIndices, [1, model.seqLen]);
            const { outputLogits } = model.forward(inputTensor);
            const lastTimeStepLogits = outputLogits.slice([0, model.seqLen - 1, 0], [1, 1, vocabData.vocabSize]).reshape([1, vocabData.vocabSize]);
            const result = getWordFromPrediction(lastTimeStepLogits, indexToWord, temperature, currentWordSequence);
            chosenWord = result.chosenWord;
            topPredictions = result.topPredictions;
        } else {
             throw new Error("Unknown model type for generation.");
        }
        
        if (chosenWord === '<unk>' || chosenWord === 'вопрос' || chosenWord === 'ответ') {
            if (chosenWord === '<unk>') break; // Stop on unknown
            continue; // Skip special tokens but continue generating
        }
        
        currentWordSequence.push(chosenWord);

        self.postMessage({
            type: 'generation-chunk',
            payload: {
                text: currentWordSequence.join(' '),
                predictions: topPredictions
            }
        });
        // Allow the event loop to process messages
        await new Promise(resolve => setTimeout(resolve, 50)); 
    }

    self.postMessage({ type: 'generation-complete' });
}


self.postMessage({ type: 'worker-ready' });

  
