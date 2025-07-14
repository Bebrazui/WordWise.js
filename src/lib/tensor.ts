
// src/lib/tensor.ts

/**
 * Класс Tensor представляет собой многомерный массив чисел с поддержкой
 * автоматического дифференцирования (autograd).
 */
export class Tensor {
  data: Float32Array; // Основные данные тензора
  shape: number[];     // Форма тензора (например, [2, 3] для матрицы 2x3)
  size: number;        // Общее количество элементов в тензоре

  grad: Tensor | null = null; // Градиент этого тензора по отношению к функции потерь
  // Список "родительских" тензоров и функций, которые вычисляют
  // локальный градиент по отношению к этому родителю (для autograd)
  _parents: { tensor: Tensor, gradFn: (grad: Tensor) => Tensor }[] = [];
  
  _isDetached: boolean = false; // Флаг для отсоединения от графа вычислений
  name: string = ''; // Имя тензора для сериализации

  /**
   * Создает новый экземпляр Tensor.
   * @param data Массив чисел или Float32Array.
   * @param shape Форма тензора.
   */
  constructor(data: number[] | Float32Array, shape: number[]) {
    this.size = shape.reduce((acc, dim) => acc * dim, 1);
    if (data.length !== this.size) {
      throw new Error(`Data length (${data.length}) does not match product of shape (${this.size}) for shape [${shape.join(',')}]`);
    }
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.shape = shape;
  }
  
  /**
   * Создает новый тензор с теми же данными, но отсоединенный от графа вычислений.
   * Это предотвращает распространение градиентов через этот тензор.
   * Используется для скрытых состояний в RNN, чтобы разорвать граф между шагами.
   * @returns Новый отсоединенный Tensor.
   */
  detach(): Tensor {
    const detachedTensor = new Tensor(this.data, this.shape);
    detachedTensor._isDetached = true;
    return detachedTensor;
  }


  /**
   * Удобный статический метод для создания тензора,
   * автоматически выводящий форму для одномерных или двумерных массивов.
   * @param data Данные тензора.
   * @param shape Опциональная форма.
   * @returns Новый Tensor.
   */
  static from(data: number[] | Float32Array, shape?: number[]): Tensor {
    if (!shape) {
      if (Array.isArray(data) && data.length > 0 && Array.isArray(data[0])) {
          shape = [data.length, (data[0] as number[]).length];
          data = (data as number[][]).flat();
      } else {
          shape = [data.length];
      }
    }
    return new Tensor(data, shape);
  }

  /**
   * Создает тензор, заполненный нулями.
   * @param shape Форма нового тензора.
   * @returns Новый Tensor, заполненный нулями.
   */
  static zeros(shape: number[]): Tensor {
    const size = shape.reduce((acc, dim) => acc * dim, 1);
    return new Tensor(new Float32Array(size).fill(0), shape);
  }

  /**
   * Создает тензор, заполненный случайными значениями из нормального распределения.
   * Использует преобразование Бокса-Мюллера.
   * @param shape Форма нового тензора.
   * @param std Стандартное отклонение для нормального распределения.
   * @returns Новый Tensor со случайными значениями.
   */
  static randn(shape: number[], std: number = 0.01): Tensor {
    const size = shape.reduce((acc, dim) => acc * dim, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        let u = 0, v = 0;
        while(u === 0) u = Math.random();
        while(v === 0) v = Math.random();
        const num = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
        data[i] = num * std;
    }
    return new Tensor(data, shape);
  }

  // --- Базовые арифметические операции ---

  /**
   * Поэлементное сложение двух тензоров. Поддерживает бродкастинг.
   * @param other Другой тензор.
   * @returns Новый Tensor с результатом сложения.
   */
    add(other: Tensor): Tensor {
        const result = new Tensor(new Float32Array(this.data), this.shape);

        // Handle scalar broadcasting
        if (other.size === 1) {
            for (let i = 0; i < result.size; i++) {
                result.data[i] += other.data[0];
            }
        }
        // Handle broadcasting of a row vector [1, N] to a matrix [M, N]
        else if (this.shape.length === 2 && other.shape.length === 2 && this.shape[0] > 1 && other.shape[0] === 1 && this.shape[1] === other.shape[1]) {
            const numRows = this.shape[0];
            const numCols = this.shape[1];
            for (let i = 0; i < numRows; i++) {
                for (let j = 0; j < numCols; j++) {
                    result.data[i * numCols + j] += other.data[j];
                }
            }
        } else { // Element-wise addition
            if (!this.shape.every((dim, i) => dim === other.shape[i])) {
                 throw new Error(`Tensors must have compatible shapes for addition (broadcasting not fully supported): [${this.shape}] vs [${other.shape}]`);
            }
            for (let i = 0; i < result.size; i++) {
                result.data[i] += other.data[i];
            }
        }

        // --- Autograd ---
        if (!this._isDetached && !other._isDetached) {
            result._parents.push(
                {
                    tensor: this,
                    gradFn: (grad) => grad
                },
                {
                    tensor: other,
                    gradFn: (grad) => {
                        if (other.size === 1) { // Gradient for scalar
                            return new Tensor([grad.data.reduce((a, b) => a + b, 0)], [1]);
                        }
                        if (other.shape.length === 2 && other.shape[0] === 1) { // Gradient for row vector
                            const biasGradData = new Float32Array(other.size).fill(0);
                            const numRows = grad.shape[0];
                            const numCols = grad.shape[1];
                            for (let i = 0; i < numRows; i++) {
                                for (let j = 0; j < numCols; j++) {
                                    biasGradData[j] += grad.data[i * numCols + j];
                                }
                            }
                            return new Tensor(biasGradData, other.shape);
                        }
                        return grad; // Gradient for element-wise
                    }
                }
            );
        }

        return result;
    }

  /**
   * Поэлементное вычитание одного тензора из другого.
   * @param other Вычитаемый тензор.
   * @returns Новый Tensor с результатом вычитания.
   */
  sub(other: Tensor): Tensor {
    if (!this.shape.every((dim, i) => dim === other.shape[i])) {
      throw new Error("Tensors must have the same shape for subtraction.");
    }
    const resultData = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      resultData[i] = this.data[i] - other.data[i];
    }
    const result = new Tensor(resultData, this.shape);
    if (!this._isDetached && !other._isDetached) {
      result._parents.push(
        { tensor: this, gradFn: (grad) => grad },
        { tensor: other, gradFn: (grad) => new Tensor(grad.data.map(g => -g), grad.shape) }
      );
    }
    return result;
  }

  /**
   * Поэлементное умножение двух тензоров. Поддерживает простейший бродкастинг со скаляром.
   * @param other Другой тензор.
   * @returns Новый Tensor с результатом умножения.
   */
  mul(other: Tensor): Tensor {
    // Простой бродкастинг для скаляров
    if (this.size === 1) {
      const resultData = new Float32Array(other.size);
      for (let i = 0; i < other.size; i++) resultData[i] = this.data[0] * other.data[i];
      const result = new Tensor(resultData, other.shape);
      if (!this._isDetached && !other._isDetached) {
        result._parents.push(
          { tensor: this, gradFn: (grad) => new Tensor([grad.data.reduce((sum, val, idx) => sum + val * other.data[idx], 0)], [1]) },
          { tensor: other, gradFn: (grad) => new Tensor(grad.data.map((g, i) => g * this.data[0]), grad.shape) }
        );
      }
      return result;
    } else if (other.size === 1) {
      const resultData = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) resultData[i] = this.data[i] * other.data[0];
      const result = new Tensor(resultData, this.shape);
       if (!this._isDetached && !other._isDetached) {
        result._parents.push(
          { tensor: this, gradFn: (grad) => new Tensor(grad.data.map((g, i) => g * other.data[0]), grad.shape) },
          { tensor: other, gradFn: (grad) => new Tensor([grad.data.reduce((sum, val, idx) => sum + val * this.data[idx], 0)], [1]) }
        );
      }
      return result;
    }

    if (!this.shape.every((dim, i) => dim === other.shape[i])) {
      throw new Error(`Tensors must have compatible shapes for element-wise multiplication: [${this.shape}] vs [${other.shape}]`);
    }
    const resultData = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      resultData[i] = this.data[i] * other.data[i];
    }
    const result = new Tensor(resultData, this.shape);
    if (!this._isDetached && !other._isDetached) {
      result._parents.push(
        { tensor: this, gradFn: (grad) => new Tensor(grad.data.map((g, i) => g * other.data[i]), grad.shape) },
        { tensor: other, gradFn: (grad) => new Tensor(grad.data.map((g, i) => g * this.data[i]), grad.shape) }
      );
    }
    return result;
  }

  /**
   * Матричное умножение (dot product) для 2D тензоров.
   * @param other Другой тензор (матрица).
   * @returns Новый Tensor с результатом матричного умножения.
   */
  dot(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error(`Dot product currently only supports 2D matrices. Got shapes: [${this.shape}] and [${other.shape}]`);
    }
    if (this.shape[1] !== other.shape[0]) {
      throw new Error(`Matrices are not compatible for dot product: ${this.shape} vs ${other.shape}`);
    }

    const m = this.shape[0]; // Строки первой матрицы
    const k = this.shape[1]; // Столбцы первой / строки второй
    const n = other.shape[1]; // Столбцы второй матрицы

    const resultData = new Float32Array(m * n).fill(0);
    const resultShape = [m, n];

    for (let i = 0; i < m; i++) { // Итерация по строкам 'this'
      for (let j = 0; j < n; j++) { // Итерация по столбцам 'other'
        let sum = 0;
        for (let l = 0; l < k; l++) { // Итерация по элементам для умножения
          sum += this.data[i * k + l] * other.data[l * n + j];
        }
        resultData[i * n + j] = sum;
      }
    }

    const result = new Tensor(resultData, resultShape);

    if (!this._isDetached && !other._isDetached) {
        // Правила для вычисления градиентов матричного умножения
        result._parents.push(
          {
            tensor: this,
            gradFn: (grad) => grad.dot(other.transpose())
          },
          {
            tensor: other,
            gradFn: (grad) => this.transpose().dot(grad)
          }
        );
    }
    return result;
  }

  /**
   * Транспонирование 2D тензора (меняет строки и столбцы местами).
   * @returns Новый транспонированный Tensor.
   */
  transpose(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error("Transpose currently only supports 2D matrices.");
    }
    const [rows, cols] = this.shape;
    const resultData = new Float32Array(this.size);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        resultData[j * rows + i] = this.data[i * cols + j];
      }
    }
    return new Tensor(resultData, [cols, rows]);
  }

  /**
   * Изменяет форму тензора без изменения данных.
   * @param newShape Новая форма.
   * @returns Новый тензор с измененной формой.
   */
  reshape(newShape: number[]): Tensor {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (this.size !== newSize) {
      throw new Error(`Cannot reshape tensor of size ${this.size} into shape [${newShape.join(',')}] with size ${newSize}`);
    }
    // Создаем новый тензор, но используем тот же буфер данных
    const result = new Tensor(this.data, newShape);
    result.name = this.name;

    if (!this._isDetached) {
      result._parents.push({
        tensor: this,
        gradFn: (grad) => grad.reshape(this.shape) // Градиент должен быть преобразован обратно в исходную форму
      });
    }
    return result;
  }

  // --- Более сложные операции (для активаций и потерь) ---

  /**
   * Применяет заданную функцию поэлементно к тензору.
   * Также принимает функцию для вычисления градиента.
   * @param fn Функция, применяемая к каждому элементу.
   * @param gradFn Функция для вычисления локального градиента.
   * @returns Новый Tensor с результатами применения функции.
   */
  apply(fn: (x: number) => number, gradFn: (x: number, y: number, g: number) => number): Tensor {
    const resultData = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      resultData[i] = fn(this.data[i]);
    }
    const result = new Tensor(resultData, this.shape);
    if (!this._isDetached) {
        result._parents.push({
          tensor: this,
          gradFn: (grad) => new Tensor(grad.data.map((g, i) => gradFn(this.data[i], resultData[i], g)), grad.shape)
        });
    }
    return result;
  }

  /**
   * Поэлементный натуральный логарифм.
   * @returns Новый Tensor с результатами логарифма.
   */
  log(): Tensor {
    return this.apply(Math.log, (x, y, g) => g / x);
  }

  /**
   * Поэлементная экспонента.
   * @returns Новый Tensor с результатами экспоненты.
   */
  exp(): Tensor {
    return this.apply(Math.exp, (x, y, g) => g * y);
  }

  /**
   * Суммирует все элементы тензора.
   * @returns Скалярный Tensor с суммой.
   */
  sum(): Tensor {
    const sumVal = this.data.reduce((acc, val) => acc + val, 0);
    const result = new Tensor([sumVal], [1]);
    if (!this._isDetached) {
        result._parents.push({
          tensor: this,
          // Градиент суммы - это единица для каждого элемента
          gradFn: (grad) => new Tensor(new Float32Array(this.size).fill(grad.data[0]), this.shape)
        });
    }
    return result;
  }

  /**
   * Вычисляет среднее значение всех элементов тензора.
   * @returns Скалярный Tensor со средним значением.
   */
  mean(): Tensor {
    const meanVal = this.data.reduce((acc, val) => acc + val, 0) / this.size;
    const result = new Tensor([meanVal], [1]);
    if (!this._isDetached) {
        result._parents.push({
          tensor: this,
          // Градиент среднего - это 1/size для каждого элемента
          gradFn: (grad) => new Tensor(new Float32Array(this.size).fill(grad.data[0] / this.size), this.shape)
        });
    }
    return result;
  }


  // --- Обратный проход (Backpropagation) ---

  /**
   * Выполняет обратный проход (backpropagation) для вычисления градиентов.
   * Градиенты обнуляются перед каждым новым вызовом.
   * @param initialGrad Опциональный начальный градиент (обычно [1.0] для функции потерь).
   */
  backward(initialGrad?: Tensor): void {
    // 1. Сброс всех градиентов в графе до начала нового backward прохода
    const nodesToReset = new Set<Tensor>();
    const stack: Tensor[] = [this];
    while (stack.length > 0) {
      const node = stack.pop()!;
      if (!nodesToReset.has(node)) {
        nodesToReset.add(node);
        node._parents.forEach(p => stack.push(p.tensor));
      }
    }
    nodesToReset.forEach(node => {
        node.grad = null;
    });

    // 2. Установка начального градиента для текущего тензора (обычно Loss)
    if (!initialGrad) {
      if (this.size !== 1) {
        throw new Error("Backward call without initialGrad expects a scalar tensor (e.g., a loss value).");
      }
      this.grad = Tensor.from([1.0]); // Градиент потери по отношению к самой себе равен 1
    } else {
      if (!this.shape.every((dim, i) => dim === initialGrad.shape[i])) {
        throw new Error(`Initial gradient shape mismatch. Expected [${this.shape}] got [${initialGrad.shape}]`);
      }
      this.grad = initialGrad;
    }

    // 3. Построение топологической сортировки графа вычислений
    const visited = new Set<Tensor>();
    const topoSort: Tensor[] = [];

    function buildTopo(node: Tensor) {
      if (!visited.has(node) && !node._isDetached) { // Не добавляем отсоединенные узлы в граф
        visited.add(node);
        node._parents.forEach(p => buildTopo(p.tensor));
        topoSort.push(node);
      }
    }
    buildTopo(this);

    // 4. Проход по графу в обратном порядке для вычисления и агрегирования градиентов
    for (let i = topoSort.length - 1; i >= 0; i--) {
      const node = topoSort[i];
      if (node.grad === null) {
          continue; // Пропускаем узлы, которые не влияют на конечную потерю
      }

      for (const p of node._parents) {
        // Вычисляем градиент для родительского тензора, используя локальный gradFn
        const upstreamGrad = p.gradFn(node.grad!);
        if (p.tensor.grad === null) {
          p.tensor.grad = upstreamGrad; // Инициализируем, если градиента еще нет
        } else {
          // Аккумулируем градиенты, если к родительскому тензору ведут несколько путей
          for (let j = 0; j < p.tensor.size; j++) {
            p.tensor.grad.data[j] += upstreamGrad.data[j];
          }
        }
      }
    }
  }

  /**
   * Возвращает строковое представление тензора для отладки.
   * @returns Строковое представление.
   */
  toString(): string {
    return `Tensor(data=[${this.data.map(d => d.toFixed(4)).join(', ')}], shape=[${this.shape.join(', ')}])`;
  }
}
