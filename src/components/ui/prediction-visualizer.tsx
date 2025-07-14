// src/components/ui/prediction-visualizer.tsx
"use client";

import { motion, AnimatePresence } from 'framer-motion';

export interface Prediction {
  word: string;
  probability: number;
}

interface PredictionVisualizerProps {
  predictions: Prediction[];
}

export function PredictionVisualizer({ predictions }: PredictionVisualizerProps) {
  if (!predictions || predictions.length === 0) {
    return null;
  }

  const maxProbability = Math.max(...predictions.map(p => p.probability), 0);

  return (
    <div className="mb-4 p-3 border rounded-lg bg-background shadow-sm">
        <h4 className="text-xs font-semibold text-muted-foreground mb-2">Монитор предсказаний (следующее слово):</h4>
        <div className="space-y-1.5">
            <AnimatePresence>
                {predictions.map((p, index) => (
                    <motion.div
                        key={p.word}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ duration: 0.3, delay: index * 0.05 }}
                        className="flex items-center gap-2 text-sm"
                    >
                        <span className="w-24 truncate font-mono text-muted-foreground" title={p.word}>{p.word}</span>
                        <div className="flex-1 bg-muted rounded-full h-4 overflow-hidden">
                            <motion.div
                                className="bg-primary/80 h-full"
                                initial={{ width: 0 }}
                                animate={{ width: `${(p.probability / maxProbability) * 100}%` }}
                                transition={{ duration: 0.5, ease: "easeOut" }}
                            />
                        </div>
                        <span className="w-12 text-right font-mono text-xs text-primary font-semibold">
                            {(p.probability * 100).toFixed(1)}%
                        </span>
                    </motion.div>
                ))}
            </AnimatePresence>
        </div>
    </div>
  );
}
