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
  
  static ones(shape: number[]): Tensor {
    const size = shape.reduce((acc, dim) => acc * dim, 1);
    return new Tensor(new Float32Array(size).fill(1), shape);
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
  
  // --- Broadcasting and Element-wise Operations ---

  private _broadcastOp(
    other: Tensor,
    op: (a: number, b: number) => number,
    thisGradFn: (grad: Tensor, a: Tensor, b: Tensor) => Tensor,
    otherGradFn: (grad: Tensor, a: Tensor, b: Tensor) => Tensor,
  ): Tensor {
      const broadcastShape = this.getBroadcastShape(this.shape, other.shape);
      const resultSize = broadcastShape.reduce((a, b) => a * b, 1);
      const resultData = new Float32Array(resultSize);
      
      const thisStrides = this.getStrides(this.shape);
      const otherStrides = this.getStrides(other.shape);
      const resultStrides = this.getStrides(broadcastShape);

      for (let i = 0; i < resultData.length; i++) {
          const indices = this.getIndices(i, resultStrides);
          const thisIndex = this.getBroadcastIndex(indices, this.shape, broadcastShape, thisStrides);
          const otherIndex = this.getBroadcastIndex(indices, other.shape, broadcastShape, otherStrides);
          resultData[i] = op(this.data[thisIndex], other.data[otherIndex]);
      }
      const result = new Tensor(resultData, broadcastShape);

      if (!this._isDetached && !other._isDetached) {
          result._parents.push({
              tensor: this,
              gradFn: (grad) => {
                  const gradForThis = thisGradFn(grad, this, other);
                  return this.sumBroadcastGrad(gradForThis, this.shape);
              }
          }, {
              tensor: other,
              gradFn: (grad) => {
                  const gradForOther = otherGradFn(grad, this, other);
                  return this.sumBroadcastGrad(gradForOther, other.shape);
              }
          });
      }
      return result;
  }

  add(other: Tensor): Tensor {
    return this._broadcastOp(
      other,
      (a, b) => a + b,
      (grad, a, b) => grad, // d/da (a+b) = 1
      (grad, a, b) => grad  // d/db (a+b) = 1
    );
  }

  sub(other: Tensor): Tensor {
      return this._broadcastOp(
          other,
          (a, b) => a - b,
          (grad, a, b) => grad, // d/da (a-b) = 1
          (grad, a, b) => grad.mul(new Tensor([-1], [1])) // d/db (a-b) = -1
      );
  }

  mul(other: Tensor): Tensor {
      return this._broadcastOp(
          other,
          (a, b) => a * b,
          (grad, a, b) => grad.mul(b), // d/da (a*b) = b
          (grad, a, b) => grad.mul(a)  // d/db (a*b) = a
      );
  }

  divScalar(scalar: number): Tensor {
        const resultData = new Float32Array(this.size);
        for (let i = 0; i < this.size; i++) {
            resultData[i] = this.data[i] / scalar;
        }
        const result = new Tensor(resultData, this.shape);
        if (!this._isDetached) {
            result._parents.push({
                tensor: this,
                gradFn: (grad) => grad.divScalar(scalar)
            });
        }
        return result;
    }

  /**
   * Матричное умножение (dot product) для 2D и 4D тензоров.
   * @param other Другой тензор (матрица).
   * @returns Новый Tensor с результатом матричного умножения.
   */
  dot(other: Tensor): Tensor {
    // Standard 2D matrix multiplication: [M, K] @ [K, N] -> [M, N]
    if (this.shape.length === 2 && other.shape.length === 2) {
        if (this.shape[1] !== other.shape[0]) {
            throw new Error(`Matrices are not compatible for dot product: ${this.shape} vs ${other.shape}`);
        }
        const [m, k] = this.shape;
        const [_, n] = other.shape;
        const resultData = new Float32Array(m * n).fill(0);

        const thisData = this.data;
        const otherData = other.data;

        for (let i = 0; i < m; i++) {
            const i_k = i * k;
            const i_n = i * n;
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let l = 0; l < k; l++) {
                    sum += thisData[i_k + l] * otherData[l * n + j];
                }
                resultData[i_n + j] = sum;
            }
        }
        const result = new Tensor(resultData, [m, n]);
        if (!this._isDetached && !other._isDetached) {
            result._parents.push({ tensor: this, gradFn: (grad) => grad.dot(other.transpose()) }, { tensor: other, gradFn: (grad) => this.transpose().dot(grad) });
        }
        return result;
    }
    // Batched matrix multiplication for models like Transformer/FlowNet output
    // [B, S, K] @ [K, N] -> [B, S, N]
    else if (this.shape.length === 3 && other.shape.length === 2) {
        if (this.shape[2] !== other.shape[0]) {
            throw new Error(`Matrices are not compatible for batched dot product: ${this.shape} vs ${other.shape}`);
        }
        const [b, s, k] = this.shape;
        const [_, n] = other.shape;
        
        const resultData = new Float32Array(b * s * n).fill(0);
        const thisData = this.data;
        const otherData = other.data;
        
        for (let bi = 0; bi < b; bi++) {
            for (let si = 0; si < s; si++) {
                const out_offset = (bi * s + si) * n;
                const in_offset = (bi * s + si) * k;
                for (let ni = 0; ni < n; ni++) {
                    let sum = 0;
                    for (let ki = 0; ki < k; ki++) {
                         sum += thisData[in_offset + ki] * otherData[ki * n + ni];
                    }
                    resultData[out_offset + ni] = sum;
                }
            }
        }
        
        const result = new Tensor(resultData, [b, s, n]);
        
        if (!this._isDetached && !other._isDetached) {
            result._parents.push(
                { tensor: this, gradFn: (grad) => grad.dot(other.transpose()) },
                { tensor: other, gradFn: (grad) => this.reshape([b * s, k]).transpose().dot(grad.reshape([b*s, n])) }
            );
        }
        return result;
    }
    // Batched 4D dot product for multi-head attention
    else if (this.shape.length === 4 && other.shape.length === 4) {
        const [b, h, n, d] = this.shape;
        const d_other = other.shape[2];
        const m = other.shape[3];

        if (d !== d_other) throw new Error(`Incompatible shapes for 4D dot product: [${this.shape}] vs [${other.shape}]`);
        
        const outShape = [b, h, n, m];
        const outData = new Float32Array(b * h * n * m).fill(0);
        
        const thisStride = [h * n * d, n * d, d, 1];
        const otherStride = [h * m * d, m * d, m, 1];
        const outStride = [h * n * m, n * m, m, 1];

        for (let bi = 0; bi < b; bi++) {
            for (let hi = 0; hi < h; hi++) {
                for (let ni = 0; ni < n; ni++) {
                    for (let mi = 0; mi < m; mi++) {
                        let sum = 0;
                        for (let di = 0; di < d; di++) {
                            const thisIdx = bi * thisStride[0] + hi * thisStride[1] + ni * thisStride[2] + di * thisStride[3];
                            const otherIdx = bi * otherStride[0] + hi * otherStride[1] + di * otherStride[2] + mi * otherStride[3];
                            sum += this.data[thisIdx] * other.data[otherIdx];
                        }
                         outData[bi * outStride[0] + hi * outStride[1] + ni * outStride[2] + mi * outStride[3]] = sum;
                    }
                }
            }
        }

        const result = new Tensor(outData, outShape);
        if (!this._isDetached && !other._isDetached) {
             result._parents.push(
                { tensor: this, gradFn: (grad) => grad.dot(other.transpose(-2, -1))},
                { tensor: other, gradFn: (grad) => this.transpose(-2, -1).dot(grad)}
             );
        }
        return result;
    }
    
    throw new Error(`Dot product not implemented for shapes: [${this.shape}] and [${other.shape}]`);
  }

  /**
   * Транспонирование N-D тензора.
   * @param axes... Оси для транспонирования. Если не указаны, транспонирует 2D матрицу.
   * @returns Новый транспонированный Tensor.
   */
  transpose(...axes: number[]): Tensor {
      if (axes.length === 0 && this.shape.length === 2) {
          axes = [1, 0];
      }
       if (axes.length !== 2 && this.shape.length !== 2) {
          throw new Error("Transpose currently requires two axes to swap for tensors with more than 2 dimensions.");
      }

      const newShape = [...this.shape];
      const [axis1, axis2] = this.getPositiveAxes(axes);
      [newShape[axis1], newShape[axis2]] = [newShape[axis2], newShape[axis1]];

      const result = Tensor.zeros(newShape);
      const originalStrides = this.getStrides(this.shape);
      const newStrides = this.getStrides(newShape);

      for(let i=0; i<this.size; i++) {
          const originalIndices = this.getIndices(i, originalStrides);
          const newIndices = [...originalIndices];
          [newIndices[axis1], newIndices[axis2]] = [newIndices[axis2], newIndices[axis1]];
          const newIndex = this.getFlatIndex(newIndices, newStrides);
          result.data[newIndex] = this.data[i];
      }

      if (!this._isDetached) {
        result._parents.push({
            tensor: this,
            gradFn: (grad) => grad.transpose(...axes)
        });
      }

      return result;
  }
  
  // Helper methods for transpose and broadcasting
  private getPositiveAxes(axes: number[]): [number, number] {
      let axis1, axis2;
       if (axes.length === 2) {
           axis1 = axes[0] < 0 ? this.shape.length + axes[0] : axes[0];
           axis2 = axes[1] < 0 ? this.shape.length + axes[1] : axes[1];
       } else { // Transpose last two dimensions by default for 2D
           axis1 = this.shape.length - 2;
           axis2 = this.shape.length - 1;
       }
      return [axis1, axis2];
  }
  private getStrides(shape: number[]): number[] {
      const strides = new Array(shape.length).fill(1);
      for(let i = shape.length - 2; i >= 0; i--) {
          strides[i] = strides[i+1] * shape[i+1];
      }
      return strides;
  }
  private getIndices(flatIndex: number, strides: number[]): number[] {
      const indices = new Array(strides.length);
      for(let i = 0; i < strides.length; i++) {
          indices[i] = Math.floor(flatIndex / strides[i]);
          flatIndex %= strides[i];
      }
      return indices;
  }
  private getFlatIndex(indices: number[], strides: number[]): number {
      let flatIndex = 0;
      for(let i = 0; i < indices.length; i++) {
          flatIndex += indices[i] * strides[i];
      }
      return flatIndex;
  }
   private getBroadcastShape(shape1: number[], shape2: number[]): number[] {
        const ndim1 = shape1.length;
        const ndim2 = shape2.length;
        const ndim = Math.max(ndim1, ndim2);
        const newShape = new Array(ndim);

        for (let i = 1; i <= ndim; i++) {
            const dim1 = i <= ndim1 ? shape1[ndim1 - i] : 1;
            const dim2 = i <= ndim2 ? shape2[ndim2 - i] : 1;
            if (dim1 !== dim2 && dim1 !== 1 && dim2 !== 1) {
                throw new Error(`Tensors cannot be broadcast together with shapes [${shape1}] and [${shape2}]`);
            }
            newShape[ndim - i] = Math.max(dim1, dim2);
        }
        return newShape;
    }

    private getBroadcastIndex(resultIndices: number[], originalShape: number[], broadcastShape: number[], originalStrides: number[]): number {
        let index = 0;
        const ndimOriginal = originalShape.length;
        const ndimBroadcast = broadcastShape.length;

        for (let i = 0; i < ndimBroadcast; i++) {
            // Align from the right
            const originalDimIndex = ndimOriginal - (ndimBroadcast - i);
            if (originalDimIndex >= 0) {
                const resultIndex = resultIndices[i];
                const originalDim = originalShape[originalDimIndex];
                // Use modulo for broadcasting dimensions of size 1
                index += (resultIndex % originalDim) * originalStrides[originalDimIndex];
            }
        }
        return index;
    }
    
    private sumBroadcastGrad(grad: Tensor, originalShape: number[]): Tensor {
        let summedGrad = grad;
        let numDimsToReduce = summedGrad.shape.length - originalShape.length;
        
        // Sum across extra leading dimensions
        if (numDimsToReduce > 0) {
            const axesToSum = Array.from({length: numDimsToReduce}, (_, i) => i);
            summedGrad = summedGrad.sum(axesToSum, false);
        }


        // Sum across broadcasted dimensions (where original shape was 1)
        for (let i = 0; i < originalShape.length; i++) {
            if (originalShape[i] === 1 && summedGrad.shape[i] > 1) {
                summedGrad = summedGrad.sum(i, true);
            }
        }
        
        return summedGrad.reshape(originalShape);
    }



  /**
   * Изменяет форму тензора без изменения данных.
   * @param newShape Новая форма.
   * @returns Новый тензор с измененной формой.
   */
  reshape(newShape: number[]): Tensor {
    // Handle -1 for inferred dimension
    if (newShape.includes(-1)) {
        const product = newShape.filter(d => d !== -1).reduce((a, b) => a * b, 1);
        if (this.size % product !== 0) {
            throw new Error(`Cannot reshape tensor of size ${this.size} into shape [${newShape.join(',')}]`);
        }
        const inferredSize = this.size / product;
        newShape = newShape.map(d => d === -1 ? inferredSize : d);
    }

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
  
  slice(begin: number[], size: number[]): Tensor {
    const resultShape = [...size];
    const resultSize = resultShape.reduce((a, b) => a * b, 1);
    const resultData = new Float32Array(resultSize);
    
    const srcStrides = this.getStrides(this.shape);
    const dstStrides = this.getStrides(resultShape);
    
    const recursiveCopy = (dim: number, srcOffset: number, dstOffset: number) => {
        if (dim === begin.length) {
            resultData[dstOffset] = this.data[srcOffset];
            return;
        }
        const currentDstStride = dstStrides[dim] || 1;
        for (let i = 0; i < size[dim]; i++) {
            recursiveCopy(dim + 1, srcOffset + (begin[dim] + i) * srcStrides[dim], dstOffset + i * currentDstStride);
        }
    };
    recursiveCopy(0, 0, 0);

    const result = new Tensor(resultData, resultShape);
    if (!this._isDetached) {
        result._parents.push({
            tensor: this,
            gradFn: (grad) => {
                const fullGrad = Tensor.zeros(this.shape);
                const gradStrides = this.getStrides(grad.shape);
                
                const recursiveAdd = (dim: number, srcOffset: number, dstOffset: number) => {
                    if (dim === begin.length) {
                        fullGrad.data[dstOffset] += grad.data[srcOffset];
                        return;
                    }
                    const currentSrcStride = gradStrides[dim] || 1;
                    for (let i = 0; i < size[dim]; i++) {
                       recursiveAdd(dim + 1, srcOffset + i * currentSrcStride, dstOffset + (begin[dim] + i) * srcStrides[dim]);
                    }
                };
                recursiveAdd(0, 0, 0);
                return fullGrad;
            }
        });
    }
    return result;
  }

  static concat(tensors: Tensor[], axis: number): Tensor {
    if (tensors.length === 0) {
      throw new Error("Cannot concat empty list of tensors.");
    }
    const firstShape = tensors[0].shape;
    const newShape = [...firstShape];
    newShape[axis] = tensors.reduce((sum, t) => sum + t.shape[axis], 0);

    const resultSize = newShape.reduce((a, b) => a * b, 1);
    const resultData = new Float32Array(resultSize);

    let offset = 0;
    const outerSize = firstShape.slice(0, axis).reduce((a, b) => a * b, 1);
    const innerSize = firstShape.slice(axis + 1).reduce((a, b) => a * b, 1);

    for(let i = 0; i < outerSize; i++) {
        for(const t of tensors) {
            const chunkSize = t.shape[axis] * innerSize;
            const srcOffset = i * chunkSize;
            resultData.set(t.data.subarray(srcOffset, srcOffset + chunkSize), offset);
            offset += chunkSize;
        }
    }

    const result = new Tensor(resultData, newShape);
    
    if (!tensors.some(t => t._isDetached)) {
        let accumulatedSize = 0;
        for (const t of tensors) {
            const start = accumulatedSize;
            const size = t.shape[axis];
            result._parents.push({
                tensor: t,
                gradFn: (grad) => {
                    const begin = new Array(grad.shape.length).fill(0);
                    begin[axis] = start;
                    const sliceSize = [...grad.shape];
                    sliceSize[axis] = size;
                    return grad.slice(begin, sliceSize);
                }
            });
            accumulatedSize += size;
        }
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
  
  pow(exponent: number): Tensor {
      return this.apply(
          (x) => Math.pow(x, exponent),
          (x, y, g) => g * exponent * Math.pow(x, exponent - 1)
      );
  }
  
  sqrt(): Tensor {
      return this.apply(
          (x) => Math.sqrt(x),
          (x, y, g) => g * 0.5 / Math.sqrt(x)
      );
  }
  
  addScalar(scalar: number): Tensor {
      return this.apply(
        (x) => x + scalar,
        (x, y, g) => g // derivative is 1
      );
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
    sum(axis: number | number[] = -1, keepDims: boolean = false): Tensor {
      if (axis === -1) {
        const sumVal = this.data.reduce((acc, val) => acc + val, 0);
        const result = new Tensor([sumVal], [1]);
        if (!this._isDetached) {
            result._parents.push({
              tensor: this,
              gradFn: (grad) => new Tensor(new Float32Array(this.size).fill(grad.data[0]), this.shape)
            });
        }
        return result;
      }
      
      const axesToSum = (Array.isArray(axis) ? axis : [axis]).map(a => a < 0 ? this.shape.length + a : a).sort((a,b) => b-a);
      let result = this;
      for (const ax of axesToSum) {
        result = result.sumAlongAxis(ax, keepDims);
      }
      return result;
    }
    
    private sumAlongAxis(axis: number, keepDims: boolean): Tensor {
      const newShape = this.shape.filter((_, i) => i !== axis);
      
      const axisDim = this.shape[axis];
      const resultSize = this.size / axisDim;
      const resultData = new Float32Array(resultSize).fill(0);
      
      const resultStrides = this.getStrides(newShape);

      for (let i = 0; i < this.size; i++) {
          const indices = this.getIndices(i, this.getStrides(this.shape));
          indices.splice(axis, 1);
          const resultIndex = this.getFlatIndex(indices, resultStrides);
          resultData[resultIndex] += this.data[i];
      }
      
      const finalShape = keepDims ? [...newShape.slice(0, axis), 1, ...newShape.slice(axis)] : newShape;
      const result = new Tensor(resultData, finalShape);

      if (!this._isDetached) {
          result._parents.push({
            tensor: this,
            gradFn: (grad) => {
                 const gradData = new Float32Array(this.size);
                 const gradShape = grad.shape;
                 
                 for(let i=0; i<this.size; i++) {
                    const indices = this.getIndices(i, this.getStrides(this.shape));
                    if(keepDims) {
                        indices[axis] = 0;
                    } else {
                        indices.splice(axis, 1);
                    }
                    const gradIndex = this.getFlatIndex(indices, this.getStrides(gradShape));
                    gradData[i] = grad.data[gradIndex];
                 }
                 return new Tensor(gradData, this.shape);
            }
          });
      }
      return result;
    }


  /**
   * Вычисляет среднее значение всех элементов тензора.
   * @returns Скалярный Tensor со средним значением.
   */
  mean(axis: number = -1, keepDims: boolean = false): Tensor {
      if (axis === -1) {
        const meanVal = this.data.reduce((acc, val) => acc + val, 0) / this.size;
        const result = new Tensor([meanVal], [1]);
        if (!this._isDetached) {
            result._parents.push({
                tensor: this,
                gradFn: (grad) => new Tensor(new Float32Array(this.size).fill(grad.data[0] / this.size), this.shape)
            });
        }
        return result;
      }
      
      const realAxis = axis < 0 ? this.shape.length + axis : axis;
      const axisDim = this.shape[realAxis];

      const sumTensor = this.sum(realAxis, keepDims);
      const result = sumTensor.divScalar(axisDim);
      
      if (!this._isDetached) {
          // The gradFn is now part of the sum and divScalar operations.
      }
      return result;
  }


  // --- Обратный проход (Backpropagation) ---

  /**
   * Выполняет обратный проход (backpropagation) для вычисления градиентов.
   * Градиенты обнуляются перед каждым новым вызовом в оптимизаторе.
   * @param initialGrad Опциональный начальный градиент (обычно [1.0] для функции потерь).
   */
  backward(initialGrad?: Tensor): void {
    // 1. Установка начального градиента для текущего тензора (обычно Loss)
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

    // 2. Построение топологической сортировки графа вычислений
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

    // 3. Проход по графу в обратном порядке для вычисления и агрегирования градиентов
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
