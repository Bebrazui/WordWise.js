
// src/lib/flownet.ts
import { Tensor } from './tensor';
import { Layer, Linear, LayerNorm, sigmoid } from './layers';

/**
 * Contextual Recurrence Unit (CRU)
 * The core of the FlowNet block, replacing the attention mechanism.
 * It maintains a persistent state that gets updated at each time step.
 */
class ContextualRecurrenceUnit extends Layer {
    linear_x: Linear;
    linear_s: Linear;
    parameters: Tensor[];

    constructor(dim: number) {
        super();
        this.linear_x = new Linear(dim, dim);
        this.linear_s = new Linear(dim, dim);
        
        this.linear_x.weights.name = 'cru_x_w'; this.linear_x.bias.name = 'cru_x_b';
        this.linear_s.weights.name = 'cru_s_w'; this.linear_s.bias.name = 'cru_s_b';

        this.parameters = [...this.linear_x.getParameters(), ...this.linear_s.getParameters()];
    }

    /**
     * @param x Input for the current time step: [B, D]
     * @param state Persistent state from the previous step: [B, D]
     * @returns The updated state for the current time step: [B, D]
     */
    forward(x: Tensor, state: Tensor): Tensor {
        // A simplified recurrent update. A real implementation might use more complex gating.
        const projected_x = this.linear_x.forward(x);
        const projected_s = this.linear_s.forward(state);
        // New state is a mix of new input and old state
        return projected_x.add(projected_s).apply(Math.tanh, (x,y,g) => g * (1 - y*y)); // tanh activation
    }
}

/**
 * The main FlowNetBlock, containing the CRU and other components.
 */
export class FlowNetBlock extends Layer {
    cru: ContextualRecurrenceUnit;
    gate: Linear; // Adaptive Gating Mechanism (simplified)
    conv: Linear; // Local Convolutional Module (simplified as a Linear layer for now)
    ffn1: Linear;
    ffn2: Linear;
    norm1: LayerNorm;
    norm2: LayerNorm;
    parameters: Tensor[];

    constructor(dim: number) {
        super();
        this.cru = new ContextualRecurrenceUnit(dim);
        this.gate = new Linear(dim, dim);
        this.conv = new Linear(dim, dim); // Simplified
        this.ffn1 = new Linear(dim, dim * 4);
        this.ffn2 = new Linear(dim * 4, dim);
        this.norm1 = new LayerNorm(dim);
        this.norm2 = new LayerNorm(dim);

        this.gate.weights.name = 'gate_w'; this.gate.bias.name = 'gate_b';
        this.conv.weights.name = 'conv_w'; this.conv.bias.name = 'conv_b';
        this.ffn1.weights.name = 'ffn1_w'; this.ffn1.bias.name = 'ffn1_b';
        this.ffn2.weights.name = 'ffn2_w'; this.ffn2.bias.name = 'ffn2_b';
        this.norm1.gamma.name = 'norm1_g'; this.norm1.beta.name = 'norm1_b';
        this.norm2.gamma.name = 'norm2_g'; this.norm2.beta.name = 'norm2_b';

        this.parameters = [
            ...this.cru.getParameters(),
            ...this.gate.getParameters(),
            ...this.conv.getParameters(),
            ...this.ffn1.getParameters(),
            ...this.ffn2.getParameters(),
            ...this.norm1.getParameters(),
            ...this.norm2.getParameters(),
        ];
    }
    
    /**
     * Processes one time step through the block.
     * @param x Input for the current time step: [B, D]
     * @param state Persistent state from the previous step: [B, D]
     * @returns An object containing the output of the block and the new state.
     */
    forward(x: Tensor, state: Tensor): { output: Tensor, newState: Tensor } {
        // --- Recurrence & Gating Part ---
        const norm_x = this.norm1.forward(x);
        const newState = this.cru.forward(norm_x, state);
        
        const g = sigmoid(this.gate.forward(norm_x));
        const local_features = this.conv.forward(norm_x);
        
        // Gated combination of recurrent state and local features
        const y = newState.mul(g).add(local_features.mul(new Tensor([1], [1]).sub(g)));
        
        // --- Feed-Forward Part ---
        const norm_y = this.norm2.forward(y);
        let ffn_out = this.ffn1.forward(norm_y);
        ffn_out = ffn_out.apply(x => Math.max(0, x), (x, y, g) => (x > 0 ? g : 0)); // ReLU
        ffn_out = this.ffn2.forward(ffn_out);
        
        const output = y.add(ffn_out);
        
        return { output, newState };
    }
}

    