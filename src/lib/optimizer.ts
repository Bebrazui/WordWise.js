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
  // Метод zeroGrad теперь не нужен, так как градиенты обнуляются
  // автоматически в Tensor.backward() перед каждым новым проходом.
}
