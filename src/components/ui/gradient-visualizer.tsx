// src/components/ui/gradient-visualizer.tsx
"use client";

import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';

export interface GradientData {
  layer: string;
  avgGrad: number;
}

interface GradientVisualizerProps {
  history: GradientData[];
}

const MAX_GRAD_VALUE = 0.1; // Heuristic max value for visualization scaling

export function GradientVisualizer({ history }: GradientVisualizerProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Пульс Градиентов</CardTitle>
        <CardDescription>
          Визуализация среднего градиента по слоям. Помогает диагностировать "затухающие" (близко к нулю) или "взрывающиеся" (очень большие) градиенты.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {history.length === 0 ? (
          <div className="flex items-center justify-center h-24 text-muted-foreground">
            Начните обучение, чтобы увидеть градиенты.
          </div>
        ) : (
          <div className="space-y-3">
            {history.map(({ layer, avgGrad }) => {
              const gradMagnitude = Math.min(Math.abs(avgGrad), MAX_GRAD_VALUE) / MAX_GRAD_VALUE;
              const barWidth = `${gradMagnitude * 100}%`;
              // Color changes from blue (low grad) to red (high grad)
              const hue = (1 - gradMagnitude) * 240; // 240 is blue, 0 is red

              return (
                <div key={layer}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm font-medium text-foreground">{layer}</span>
                    <span className="text-xs font-mono text-muted-foreground">
                      {avgGrad.toExponential(2)}
                    </span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-2.5">
                    <motion.div
                      className="h-2.5 rounded-full"
                      style={{ 
                        width: barWidth, 
                        // Using a simple color transition for effect
                        background: `linear-gradient(90deg, hsl(210, 80%, 60%), hsl(${hue}, 80%, 60%))`
                      }}
                      initial={{ width: 0 }}
                      animate={{ width: barWidth }}
                      transition={{ duration: 0.5, ease: 'easeOut' }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
