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
        await loadModel(payload.modelJson);
        break;
      case 'generate':
        await generate(payload);
        break;
    }
  } catch (error) {
    self.postMessage({ type: 'error', payload: { message: error instanceof Error ? error.message : String(error), error } });
  }
};


async function loadModel(modelJson: string) {
    const loaded = deserializeModel(modelJson);
    model = loaded.model;
    vocabData = loaded.vocabData;
    
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

  // Common data preparation for all text models
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
    throw new Error('Model is not initialized or no training data is available.');
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
  
  if (model.stopTraining) {
       const lastEpoch = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].epoch : 0;
       self.postMessage({ type: 'training-stopped', payload: { epoch: lastEpoch } });
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

    let currentWord = startWord.toLowerCase();
    if (!wordToIndex.has(currentWord)) {
        currentWord = '<unk>';
    }

    let generatedSequence = [currentWord];
    let finalPredictions: any[] = [];

    if (model instanceof WordWiseModel) {
        let {h0: h, c0: c} = model.initializeStates(1);
        for (let i = 0; i < numWords; i++) {
            const inputTensor = new Tensor([wordToIndex.get(currentWord) || 0], [1]);
            const { outputLogits, h: nextH, c: nextC } = model.forward(inputTensor, h, c);
            h = nextH.detach(); c = nextC.detach();
            const { chosenWord, topPredictions } = getWordFromPrediction(outputLogits, indexToWord, temperature, generatedSequence);
            finalPredictions = topPredictions;
            if (chosenWord === '<unk>') break;
            generatedSequence.push(chosenWord);
            currentWord = chosenWord;
        }
    } else if (model instanceof TransformerModel || model instanceof FlowNetModel) {
         for (let i = 0; i < numWords; i++) {
            const currentSequenceIndices = generatedSequence.slice(-model.seqLen).map(w => wordToIndex.get(w) || 0);
            // Pad if necessary
            while(currentSequenceIndices.length < model.seqLen) {
                currentSequenceIndices.unshift(0); // Pad with <unk>
            }
            const inputTensor = new Tensor(currentSequenceIndices, [1, model.seqLen]);
            const { outputLogits } = model.forward(inputTensor); // [1, seqLen, vocabSize]
            // We only care about the prediction for the very last token in the sequence
            const lastTimeStepLogits = outputLogits.slice([0, model.seqLen - 1, 0], [1, 1, vocabData.vocabSize]).reshape([1, vocabData.vocabSize]);
            const { chosenWord, topPredictions } = getWordFromPrediction(lastTimeStepLogits, indexToWord, temperature, generatedSequence);
            finalPredictions = topPredictions;
            if (chosenWord === '<unk>') break;
            generatedSequence.push(chosenWord);
            currentWord = chosenWord;
         }
    }

    self.postMessage({
        type: 'generation-result',
        payload: {
            text: generatedSequence.join(' '),
            predictions: finalPredictions
        }
    });
}


self.postMessage({ type: 'worker-ready' });

    