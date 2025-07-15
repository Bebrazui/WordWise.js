
import { create } from 'zustand';
import type { AnyModel, VocabData } from '@/lib/model';

type TrainedModelState = {
  modelJson: string | null; // Store the serialized model as a JSON string
  temperature: number;
  setModelJson: (json: string | null) => void;
  setTemperature: (temp: number) => void;
};

export const useTrainedModel = create<TrainedModelState>((set) => ({
  modelJson: null,
  temperature: 0.8,
  setModelJson: (json) => set({ modelJson: json }),
  setTemperature: (temp) => set({ temperature: temp }),
}));
