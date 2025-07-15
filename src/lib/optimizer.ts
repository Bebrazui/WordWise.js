// src/lib/optimizer.ts
import { Tensor } from './tensor';

/**
 * Оптимизатор Стохастического Градиентного Спуска (SGD).
 * Обновляет параметры модели, вычитая из них градиент, умноженный на скорость обучения.
 */
export class SGD {
  learningRate: number; // Скорость обучения

  /**
   * @param learningRate Скорость обучения (по умолчанию 0.01).
   */
  constructor(learningRate: number = 0.01) {
    this.learningRate = learningRate;
  }
  
  /**
   * Обнуляет градиенты для всех предоставленных параметров.
   * Это нужно вызывать перед каждым вызовом backward().
   * @param parameters Массив обучаемых тензоров.
   */
  zeroGrad(parameters: Tensor[]): void {
      parameters.forEach(p => {
          if (p.grad) {
              p.grad = null;
          }
      });
  }

  /**
   * Выполняет один шаг оптимизации: обновляет веса всех предоставленных параметров.
   * @param parameters Массив обучаемых тензоров (весов и смещений модели).
   */
  step(parameters: Tensor[]): void {
    parameters.forEach(param => {
      // Обновляем только те параметры, у которых есть градиент
      if (param.grad) {
        for (let i = 0; i < param.size; i++) {
          param.data[i] -= this.learningRate * param.grad.data[i];
        }
      }
    });
  }
}
