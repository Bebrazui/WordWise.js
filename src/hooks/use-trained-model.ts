
import { create } from 'zustand';
import type { WordWiseModel } from '@/lib/model';

type VocabData = {
  vocab: string[];
  wordToIndex: Map<string, number>;
  indexToWord: Map<number, string>;
  vocabSize: number;
};

type TrainedModelState = {
  trainedModel: WordWiseModel | null;
  vocabData: VocabData | null;
  setTrainedModel: (model: WordWiseModel | null) => void;
  setVocabData: (data: VocabData | null) => void;
};

export const useTrainedModel = create<TrainedModelState>((set) => ({
  trainedModel: null,
  vocabData: null,
  setTrainedModel: (model) => set({ trainedModel: model }),
  setVocabData: (data) => set({ vocabData: data }),
}));
