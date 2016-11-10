#include "matrix.hpp"
#include <vector>

#ifndef __LSTMSTATESTRUCT__
#define __LSTMSTATESTRUCT__
template<unsigned long input_size, unsigned long mem_cell_size>
struct LstmState
{
    static constexpr unsigned long concat_size=input_size+mem_cell_size;
    //LSTM states of inputs+h after passed through weights (synapses) and activation function applied to them.
    Matrix<1,mem_cell_size> state_g;
    Matrix<1,mem_cell_size> state_i;
    Matrix<1,mem_cell_size> state_f;
    Matrix<1,mem_cell_size> state_o;
    Matrix<1,mem_cell_size> state_s;
    //Further internal states
    Matrix<1,mem_cell_size> state_st;
    Matrix<1,mem_cell_size> state_h;
    //Deltas
    Matrix<1,mem_cell_size> delta_h;
    Matrix<1,mem_cell_size> delta_o;
    Matrix<1,mem_cell_size> delta_s;
    Matrix<1,mem_cell_size> delta_i;
    Matrix<1,mem_cell_size> delta_g;
    Matrix<1,mem_cell_size> delta_f;
    //These two deltas are used by the next timestep (backpropagation through time)
    Matrix<1,mem_cell_size> delta_ls;//last s
    Matrix<1,mem_cell_size> delta_lh;//last h
};
#endif

template<unsigned long input_size, unsigned long mem_cell_size>
class BaseLSTMBlock
{
protected:
    std::vector<LstmState<input_size, mem_cell_size>> lstm_states;
    static constexpr unsigned long concat_size=input_size+mem_cell_size;
    //LSTM weights
    Matrix<input_size, mem_cell_size> weights_xg;
    Matrix<input_size, mem_cell_size> weights_xi;
    Matrix<input_size, mem_cell_size> weights_xf;
    Matrix<input_size, mem_cell_size> weights_xo;
    Matrix<mem_cell_size, mem_cell_size> weights_hg;
    Matrix<mem_cell_size, mem_cell_size> weights_hi;
    Matrix<mem_cell_size, mem_cell_size> weights_hf;
    Matrix<mem_cell_size, mem_cell_size> weights_ho;
    //LSTM biases
    Matrix<1, mem_cell_size> bias_g;
    Matrix<1, mem_cell_size> bias_i;
    Matrix<1, mem_cell_size> bias_f;
    Matrix<1, mem_cell_size> bias_o;

    //LSTM gradient accumulator for weights
    Matrix<input_size, mem_cell_size> weights_xg_gradient_acc;
    Matrix<input_size, mem_cell_size> weights_xi_gradient_acc;
    Matrix<input_size, mem_cell_size> weights_xf_gradient_acc;
    Matrix<input_size, mem_cell_size> weights_xo_gradient_acc;
    Matrix<mem_cell_size, mem_cell_size> weights_hg_gradient_acc;
    Matrix<mem_cell_size, mem_cell_size> weights_hi_gradient_acc;
    Matrix<mem_cell_size, mem_cell_size> weights_hf_gradient_acc;
    Matrix<mem_cell_size, mem_cell_size> weights_ho_gradient_acc;
    //LSTM gradient accumulator for biases
    Matrix<1, mem_cell_size> bias_g_gradient_acc;
    Matrix<1, mem_cell_size> bias_i_gradient_acc;
    Matrix<1, mem_cell_size> bias_f_gradient_acc;
    Matrix<1, mem_cell_size> bias_o_gradient_acc;
public:
    //Constructor: randomizes the weights and biases
    BaseLSTMBlock(size_t time_steps=0) noexcept:lstm_states(time_steps)
    , weights_xg_gradient_acc(0.0), weights_xi_gradient_acc(0.0), weights_xf_gradient_acc(0.0), weights_xo_gradient_acc(0.0)
    , weights_hg_gradient_acc(0.0), weights_hi_gradient_acc(0.0), weights_hf_gradient_acc(0.0), weights_ho_gradient_acc(0.0)
    , bias_g_gradient_acc(0.0), bias_i_gradient_acc(0.0), bias_f_gradient_acc(0.0), bias_o_gradient_acc(0.0)
    {
        weights_xg.randomize_for_nn(concat_size+1);
        weights_xi.randomize_for_nn(concat_size+1);
        weights_xf.randomize_for_nn(concat_size+1);
        weights_xo.randomize_for_nn(concat_size+1);
        weights_hg.randomize_for_nn(concat_size+1);
        weights_hi.randomize_for_nn(concat_size+1);
        weights_hf.randomize_for_nn(concat_size+1);
        weights_ho.randomize_for_nn(concat_size+1);
        bias_g.randomize_for_nn(concat_size+1);
        bias_i.randomize_for_nn(concat_size+1);
        bias_f.randomize_for_nn(concat_size+1);
        bias_o.randomize_for_nn(concat_size+1);
    }

    inline void only_wb_to_bin_file(std::ofstream &out) noexcept
    {
        weights_xg.to_bin_file(out);
        weights_xi.to_bin_file(out);
        weights_xf.to_bin_file(out);
        weights_xo.to_bin_file(out);
        weights_hg.to_bin_file(out);
        weights_hi.to_bin_file(out);
        weights_hf.to_bin_file(out);
        weights_ho.to_bin_file(out);
        bias_g.to_bin_file(out);
        bias_i.to_bin_file(out);
        bias_f.to_bin_file(out);
        bias_o.to_bin_file(out);
    }

    inline void only_wb_from_bin_file(std::ifstream &in) noexcept
    {
        weights_xg.from_bin_file(in);
        weights_xi.from_bin_file(in);
        weights_xf.from_bin_file(in);
        weights_xo.from_bin_file(in);
        weights_hg.from_bin_file(in);
        weights_hi.from_bin_file(in);
        weights_hf.from_bin_file(in);
        weights_ho.from_bin_file(in);
        bias_g.from_bin_file(in);
        bias_i.from_bin_file(in);
        bias_f.from_bin_file(in);
        bias_o.from_bin_file(in);
    }

    inline void set_time_steps(size_t time_steps) noexcept
    {
        lstm_states.resize(time_steps);
    }

    inline void reserve_time_steps(size_t time_steps) noexcept
    {
        lstm_states.reserve(time_steps);
    }

    inline void calc(const Matrix<1,input_size> &X, size_t time_step) noexcept
    {
        assert(time_step<lstm_states.size());
        //Calculate states g, i (input gate), f (forget gate), and o (output gate). steps are split for readability
        //f does not need to be calculated in the first round (nothing to forget there). maybe optimize that later.

        //Multiply input with corresponding weights for each state
        lstm_states[time_step].state_g.equals_a_dot_b(X,weights_xg);
        lstm_states[time_step].state_i.equals_a_dot_b(X,weights_xi);
        lstm_states[time_step].state_f.equals_a_dot_b(X,weights_xf);
        lstm_states[time_step].state_o.equals_a_dot_b(X,weights_xo);
        if(time_step!=0)
        {
            //Multiply last h-state with corresponding weights for each state
            lstm_states[time_step].state_g.add_a_dot_b(lstm_states[time_step-1].state_h, weights_hg);
            lstm_states[time_step].state_i.add_a_dot_b(lstm_states[time_step-1].state_h, weights_hi);
            lstm_states[time_step].state_f.add_a_dot_b(lstm_states[time_step-1].state_h, weights_hf);
            lstm_states[time_step].state_o.add_a_dot_b(lstm_states[time_step-1].state_h, weights_ho);
        }
        //Add biases to each state
        lstm_states[time_step].state_g.add(bias_g);
        lstm_states[time_step].state_i.add(bias_i);
        lstm_states[time_step].state_f.add(bias_f);
        lstm_states[time_step].state_o.add(bias_o);
        //Apply activation function to each state
        lstm_states[time_step].state_g.apply_tanh();
        lstm_states[time_step].state_i.apply_sigmoid();
        lstm_states[time_step].state_f.apply_sigmoid();
        lstm_states[time_step].state_o.apply_sigmoid();

        //Calculate s-state. This is the "memory state" which passes information to subsecuent timesteps
        if(time_step!=0)lstm_states[time_step].state_s.equals_a_mul_b_add_c_mul_d(lstm_states[time_step].state_g, lstm_states[time_step].state_i, lstm_states[time_step-1].state_s, lstm_states[time_step].state_f);
        else lstm_states[time_step].state_s.equals_a_mul_b(lstm_states[time_step].state_g, lstm_states[time_step].state_i);

        //The "memory state" s needs to have a element-wise tanh function applied to it for further calculations
        lstm_states[time_step].state_st.set(lstm_states[time_step].state_s);
        lstm_states[time_step].state_st.apply_tanh();

        //Calculate the output of the LSTM (tanh of output of mem-cell times output gate)
        lstm_states[time_step].state_h.equals_a_mul_b(lstm_states[time_step].state_st, lstm_states[time_step].state_o);
    }

    inline void set_first_delta(const Matrix<1,mem_cell_size> &Y, size_t time_step) noexcept
    {
        assert(time_step<lstm_states.size());
        //Get outputs delta
        lstm_states[time_step].delta_h.equals_a_sub_b(Y,lstm_states[time_step].state_h);
    }

    inline void propagate_delta(size_t time_step, size_t total_time_steps) noexcept
    {
        assert(time_step<lstm_states.size() && time_step<total_time_steps);
        //Add deltas from future timesteps to the h-state delta
        if(time_step<total_time_steps-1)lstm_states[time_step].delta_h.add(lstm_states[time_step+1].delta_lh);
        //Get delta of the o-state
        lstm_states[time_step].delta_o.equals_a_mul_b(lstm_states[time_step].delta_h, lstm_states[time_step].state_st);
        lstm_states[time_step].delta_o.mult_after_func01(lstm_states[time_step].state_o);
        //Get delta of the s-state
        lstm_states[time_step].delta_s.equals_a_mul_b(lstm_states[time_step].delta_h, lstm_states[time_step].state_o);
        lstm_states[time_step].delta_s.mult_after_func02(lstm_states[time_step].state_st);
        if(time_step!=total_time_steps-1)lstm_states[time_step].delta_s.add(lstm_states[time_step+1].delta_ls);

        //Get delta of the i-state
        lstm_states[time_step].delta_i.equals_a_mul_b(lstm_states[time_step].delta_s, lstm_states[time_step].state_g);
        lstm_states[time_step].delta_i.mult_after_func01(lstm_states[time_step].state_i);
        //Get delta of the g-state
        lstm_states[time_step].delta_g.equals_a_mul_b(lstm_states[time_step].delta_s, lstm_states[time_step].state_i);
        lstm_states[time_step].delta_g.mult_after_func02(lstm_states[time_step].state_g);
        //Get delta of the f-state and last s-state //f state does not exist in the first round anyways
        //Both deltas are not needed in the first round, so they are not calculated
        if(time_step!=0)
        {
            lstm_states[time_step].delta_f.equals_a_mul_b(lstm_states[time_step].delta_s, lstm_states[time_step-1].state_s);
            lstm_states[time_step].delta_f.mult_after_func01(lstm_states[time_step].state_f);
            lstm_states[time_step].delta_ls.equals_a_mul_b(lstm_states[time_step].delta_s, lstm_states[time_step].state_f);

            lstm_states[time_step].delta_lh.equals_a_dot_bt(lstm_states[time_step].delta_i, weights_hi);
            lstm_states[time_step].delta_lh.add_a_dot_bt(lstm_states[time_step].delta_f, weights_hf);
            lstm_states[time_step].delta_lh.add_a_dot_bt(lstm_states[time_step].delta_o, weights_ho);
            lstm_states[time_step].delta_lh.add_a_dot_bt(lstm_states[time_step].delta_g, weights_hg);
        }
    }

    inline void propagate_delta(Matrix<1,input_size> &X_delta, size_t time_step, size_t total_time_steps) noexcept
    {
        propagate_delta(time_step, total_time_steps);

        X_delta.equals_a_dot_bt(lstm_states[time_step].delta_i, weights_xi);
        X_delta.add_a_dot_bt(lstm_states[time_step].delta_f, weights_xf);
        X_delta.add_a_dot_bt(lstm_states[time_step].delta_o, weights_xo);
        X_delta.add_a_dot_bt(lstm_states[time_step].delta_g, weights_xg);
    }

    inline void accumulate_gradients(const Matrix<1,input_size> &X, size_t time_step) noexcept
    {
        assert(time_step<lstm_states.size());

        weights_xg_gradient_acc.add_at_dot_b(X, lstm_states[time_step].delta_g);
        weights_xi_gradient_acc.add_at_dot_b(X, lstm_states[time_step].delta_i);
        weights_xf_gradient_acc.add_at_dot_b(X, lstm_states[time_step].delta_f);
        weights_xo_gradient_acc.add_at_dot_b(X, lstm_states[time_step].delta_o);
        if(time_step!=0)
        {
            weights_hg_gradient_acc.add_at_dot_b(lstm_states[time_step-1].state_h, lstm_states[time_step].delta_g);
            weights_hi_gradient_acc.add_at_dot_b(lstm_states[time_step-1].state_h, lstm_states[time_step].delta_i);
            weights_hf_gradient_acc.add_at_dot_b(lstm_states[time_step-1].state_h, lstm_states[time_step].delta_f);
            weights_ho_gradient_acc.add_at_dot_b(lstm_states[time_step-1].state_h, lstm_states[time_step].delta_o);
        }
        bias_g_gradient_acc.add(lstm_states[time_step].delta_g);
        bias_i_gradient_acc.add(lstm_states[time_step].delta_i);
        bias_f_gradient_acc.add(lstm_states[time_step].delta_f);
        bias_o_gradient_acc.add(lstm_states[time_step].delta_o);
    }

    inline const Matrix<1,mem_cell_size>& get_output(size_t time_step) const noexcept
    {
        return lstm_states[time_step].state_h;
    }

    inline Matrix<1,mem_cell_size>& get_delta_output(size_t time_step) noexcept
    {
        return lstm_states[time_step].delta_h;
    }
};

template<unsigned long input_size, unsigned long mem_cell_size>
class NAGLSTMBlock : public BaseLSTMBlock<input_size, mem_cell_size>
{
private:
    //LSTM momentum for weights
    Matrix<input_size, mem_cell_size> moment_weights_xg;
    Matrix<input_size, mem_cell_size> moment_weights_xi;
    Matrix<input_size, mem_cell_size> moment_weights_xf;
    Matrix<input_size, mem_cell_size> moment_weights_xo;
    Matrix<mem_cell_size, mem_cell_size> moment_weights_hg;
    Matrix<mem_cell_size, mem_cell_size> moment_weights_hi;
    Matrix<mem_cell_size, mem_cell_size> moment_weights_hf;
    Matrix<mem_cell_size, mem_cell_size> moment_weights_ho;
    //LSTM momentum for biases
    Matrix<1, mem_cell_size> moment_bias_g;
    Matrix<1, mem_cell_size> moment_bias_i;
    Matrix<1, mem_cell_size> moment_bias_f;
    Matrix<1, mem_cell_size> moment_bias_o;
public:
    using BaseLSTMBlock<input_size,mem_cell_size>::lstm_states;

    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xg;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xi;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xf;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xo;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hg;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hi;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hf;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_ho;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_g;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_i;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_f;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_o;

    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xg_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xi_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xf_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xo_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hg_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hi_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hf_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_ho_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_g_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_i_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_f_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_o_gradient_acc;
    NAGLSTMBlock(size_t time_steps=0) noexcept:BaseLSTMBlock<input_size,mem_cell_size>(time_steps)
    , moment_weights_xg(0.0), moment_weights_xi(0.0), moment_weights_xf(0.0), moment_weights_xo(0.0)
    , moment_weights_hg(0.0), moment_weights_hi(0.0), moment_weights_hf(0.0), moment_weights_ho(0.0)
    , moment_bias_g(0.0), moment_bias_i(0.0), moment_bias_f(0.0), moment_bias_o(0.0)
    {
    }

    inline void reset_momentum() noexcept
    {
        moment_weights_xg.set(0.0);
        moment_weights_xi.set(0.0);
        moment_weights_xf.set(0.0);
        moment_weights_xo.set(0.0);
        moment_weights_hg.set(0.0);
        moment_weights_hi.set(0.0);
        moment_weights_hf.set(0.0);
        moment_weights_ho.set(0.0);
        moment_bias_g.set(0.0);
        moment_bias_i.set(0.0);
        moment_bias_f.set(0.0);
        moment_bias_o.set(0.0);
    }

    inline void to_file(std::ofstream &out) noexcept
    {
        weights_xg.to_file(out);
        weights_xi.to_file(out);
        weights_xf.to_file(out);
        weights_xo.to_file(out);
        weights_hg.to_file(out);
        weights_hi.to_file(out);
        weights_hf.to_file(out);
        weights_ho.to_file(out);
        bias_g.to_file(out);
        bias_i.to_file(out);
        bias_f.to_file(out);
        bias_o.to_file(out);

        moment_weights_xg.to_file(out);
        moment_weights_xi.to_file(out);
        moment_weights_xf.to_file(out);
        moment_weights_xo.to_file(out);
        moment_weights_hg.to_file(out);
        moment_weights_hi.to_file(out);
        moment_weights_hf.to_file(out);
        moment_weights_ho.to_file(out);
        moment_bias_g.to_file(out);
        moment_bias_i.to_file(out);
        moment_bias_f.to_file(out);
        moment_bias_o.to_file(out);
    }

    inline void from_file(std::ifstream &in) noexcept
    {
        weights_xg.from_file(in);
        weights_xi.from_file(in);
        weights_xf.from_file(in);
        weights_xo.from_file(in);
        weights_hg.from_file(in);
        weights_hi.from_file(in);
        weights_hf.from_file(in);
        weights_ho.from_file(in);
        bias_g.from_file(in);
        bias_i.from_file(in);
        bias_f.from_file(in);
        bias_o.from_file(in);

        moment_weights_xg.from_file(in);
        moment_weights_xi.from_file(in);
        moment_weights_xf.from_file(in);
        moment_weights_xo.from_file(in);
        moment_weights_hg.from_file(in);
        moment_weights_hi.from_file(in);
        moment_weights_hf.from_file(in);
        moment_weights_ho.from_file(in);
        moment_bias_g.from_file(in);
        moment_bias_i.from_file(in);
        moment_bias_f.from_file(in);
        moment_bias_o.from_file(in);
    }

    inline void to_bin_file(std::ofstream &out) noexcept
    {
        weights_xg.to_bin_file(out);
        weights_xi.to_bin_file(out);
        weights_xf.to_bin_file(out);
        weights_xo.to_bin_file(out);
        weights_hg.to_bin_file(out);
        weights_hi.to_bin_file(out);
        weights_hf.to_bin_file(out);
        weights_ho.to_bin_file(out);
        bias_g.to_bin_file(out);
        bias_i.to_bin_file(out);
        bias_f.to_bin_file(out);
        bias_o.to_bin_file(out);

        moment_weights_xg.to_bin_file(out);
        moment_weights_xi.to_bin_file(out);
        moment_weights_xf.to_bin_file(out);
        moment_weights_xo.to_bin_file(out);
        moment_weights_hg.to_bin_file(out);
        moment_weights_hi.to_bin_file(out);
        moment_weights_hf.to_bin_file(out);
        moment_weights_ho.to_bin_file(out);
        moment_bias_g.to_bin_file(out);
        moment_bias_i.to_bin_file(out);
        moment_bias_f.to_bin_file(out);
        moment_bias_o.to_bin_file(out);
    }

    inline void from_bin_file(std::ifstream &in) noexcept
    {
        weights_xg.from_bin_file(in);
        weights_xi.from_bin_file(in);
        weights_xf.from_bin_file(in);
        weights_xo.from_bin_file(in);
        weights_hg.from_bin_file(in);
        weights_hi.from_bin_file(in);
        weights_hf.from_bin_file(in);
        weights_ho.from_bin_file(in);
        bias_g.from_bin_file(in);
        bias_i.from_bin_file(in);
        bias_f.from_bin_file(in);
        bias_o.from_bin_file(in);

        moment_weights_xg.from_bin_file(in);
        moment_weights_xi.from_bin_file(in);
        moment_weights_xf.from_bin_file(in);
        moment_weights_xo.from_bin_file(in);
        moment_weights_hg.from_bin_file(in);
        moment_weights_hi.from_bin_file(in);
        moment_weights_hf.from_bin_file(in);
        moment_weights_ho.from_bin_file(in);
        moment_bias_g.from_bin_file(in);
        moment_bias_i.from_bin_file(in);
        moment_bias_f.from_bin_file(in);
        moment_bias_o.from_bin_file(in);
    }

    inline void apply_momentum(const double momentum) noexcept
    {
        moment_weights_xg.mul(momentum);
        moment_weights_xi.mul(momentum);
        moment_weights_xf.mul(momentum);
        moment_weights_xo.mul(momentum);
        moment_weights_hg.mul(momentum);
        moment_weights_hi.mul(momentum);
        moment_weights_hf.mul(momentum);
        moment_weights_ho.mul(momentum);
        moment_bias_g.mul(momentum);
        moment_bias_i.mul(momentum);
        moment_bias_f.mul(momentum);
        moment_bias_o.mul(momentum);
        weights_xg.add(moment_weights_xg);
        weights_xi.add(moment_weights_xi);
        weights_xf.add(moment_weights_xf);
        weights_xo.add(moment_weights_xo);
        weights_hg.add(moment_weights_hg);
        weights_hi.add(moment_weights_hi);
        weights_hf.add(moment_weights_hf);
        weights_ho.add(moment_weights_ho);
        bias_g.add(moment_bias_g);
        bias_i.add(moment_bias_i);
        bias_f.add(moment_bias_f);
        bias_o.add(moment_bias_o);
    }

    inline void update_weights_momentum(const double learning_rate) noexcept
    {
        update_weight_momentum(weights_xg, moment_weights_xg, weights_xg_gradient_acc, learning_rate);
        update_weight_momentum(weights_xi, moment_weights_xi, weights_xi_gradient_acc, learning_rate);
        update_weight_momentum(weights_xf, moment_weights_xf, weights_xf_gradient_acc, learning_rate);
        update_weight_momentum(weights_xo, moment_weights_xo, weights_xo_gradient_acc, learning_rate);
        update_weight_momentum(weights_hg, moment_weights_hg, weights_hg_gradient_acc, learning_rate);
        update_weight_momentum(weights_hi, moment_weights_hi, weights_hi_gradient_acc, learning_rate);
        update_weight_momentum(weights_hf, moment_weights_hf, weights_hf_gradient_acc, learning_rate);
        update_weight_momentum(weights_ho, moment_weights_ho, weights_ho_gradient_acc, learning_rate);
        update_weight_momentum(bias_g, moment_bias_g, bias_g_gradient_acc, learning_rate);
        update_weight_momentum(bias_i, moment_bias_i, bias_i_gradient_acc, learning_rate);
        update_weight_momentum(bias_f, moment_bias_f, bias_f_gradient_acc, learning_rate);
        update_weight_momentum(bias_o, moment_bias_o, bias_o_gradient_acc, learning_rate);

        weights_xg_gradient_acc.set(0.0);
        weights_xi_gradient_acc.set(0.0);
        weights_xf_gradient_acc.set(0.0);
        weights_xo_gradient_acc.set(0.0);
        weights_hg_gradient_acc.set(0.0);
        weights_hi_gradient_acc.set(0.0);
        weights_hf_gradient_acc.set(0.0);
        weights_ho_gradient_acc.set(0.0);
        bias_g_gradient_acc.set(0.0);
        bias_i_gradient_acc.set(0.0);
        bias_f_gradient_acc.set(0.0);
        bias_o_gradient_acc.set(0.0);
    }
};

template<unsigned long input_size, unsigned long mem_cell_size>
class SpeedyLSTMBlock : public BaseLSTMBlock<input_size, mem_cell_size>
{
private:
    //LSTM momentum for weights
    Matrix<input_size, mem_cell_size> moment_weights_xg;
    Matrix<input_size, mem_cell_size> moment_weights_xi;
    Matrix<input_size, mem_cell_size> moment_weights_xf;
    Matrix<input_size, mem_cell_size> moment_weights_xo;
    Matrix<mem_cell_size, mem_cell_size> moment_weights_hg;
    Matrix<mem_cell_size, mem_cell_size> moment_weights_hi;
    Matrix<mem_cell_size, mem_cell_size> moment_weights_hf;
    Matrix<mem_cell_size, mem_cell_size> moment_weights_ho;
    //LSTM momentum for biases
    Matrix<1, mem_cell_size> moment_bias_g;
    Matrix<1, mem_cell_size> moment_bias_i;
    Matrix<1, mem_cell_size> moment_bias_f;
    Matrix<1, mem_cell_size> moment_bias_o;

    //LSTM ms for weights
    Matrix<input_size, mem_cell_size> ms_weights_xg;
    Matrix<input_size, mem_cell_size> ms_weights_xi;
    Matrix<input_size, mem_cell_size> ms_weights_xf;
    Matrix<input_size, mem_cell_size> ms_weights_xo;
    Matrix<mem_cell_size, mem_cell_size> ms_weights_hg;
    Matrix<mem_cell_size, mem_cell_size> ms_weights_hi;
    Matrix<mem_cell_size, mem_cell_size> ms_weights_hf;
    Matrix<mem_cell_size, mem_cell_size> ms_weights_ho;
    //LSTM ms for biases
    Matrix<1, mem_cell_size> ms_bias_g;
    Matrix<1, mem_cell_size> ms_bias_i;
    Matrix<1, mem_cell_size> ms_bias_f;
    Matrix<1, mem_cell_size> ms_bias_o;
public:
    using BaseLSTMBlock<input_size,mem_cell_size>::lstm_states;

    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xg;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xi;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xf;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xo;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hg;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hi;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hf;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_ho;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_g;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_i;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_f;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_o;

    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xg_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xi_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xf_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xo_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hg_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hi_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hf_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_ho_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_g_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_i_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_f_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_o_gradient_acc;
    SpeedyLSTMBlock(size_t time_steps=0) noexcept:BaseLSTMBlock<input_size,mem_cell_size>(time_steps)
    , moment_weights_xg(0.0), moment_weights_xi(0.0), moment_weights_xf(0.0), moment_weights_xo(0.0)
    , moment_weights_hg(0.0), moment_weights_hi(0.0), moment_weights_hf(0.0), moment_weights_ho(0.0)
    , moment_bias_g(0.0), moment_bias_i(0.0), moment_bias_f(0.0), moment_bias_o(0.0)
    , ms_weights_xg(1.0), ms_weights_xi(1.0), ms_weights_xf(1.0), ms_weights_xo(1.0)
    , ms_weights_hg(1.0), ms_weights_hi(1.0), ms_weights_hf(1.0), ms_weights_ho(1.0)
    , ms_bias_g(1.0), ms_bias_i(1.0), ms_bias_f(1.0), ms_bias_o(1.0)
    {
    }

    inline void to_file(std::ofstream &out) noexcept
    {
        weights_xg.to_file(out);
        weights_xi.to_file(out);
        weights_xf.to_file(out);
        weights_xo.to_file(out);
        weights_hg.to_file(out);
        weights_hi.to_file(out);
        weights_hf.to_file(out);
        weights_ho.to_file(out);
        bias_g.to_file(out);
        bias_i.to_file(out);
        bias_f.to_file(out);
        bias_o.to_file(out);

        moment_weights_xg.to_file(out);
        moment_weights_xi.to_file(out);
        moment_weights_xf.to_file(out);
        moment_weights_xo.to_file(out);
        moment_weights_hg.to_file(out);
        moment_weights_hi.to_file(out);
        moment_weights_hf.to_file(out);
        moment_weights_ho.to_file(out);
        moment_bias_g.to_file(out);
        moment_bias_i.to_file(out);
        moment_bias_f.to_file(out);
        moment_bias_o.to_file(out);

        ms_weights_xg.to_file(out);
        ms_weights_xi.to_file(out);
        ms_weights_xf.to_file(out);
        ms_weights_xo.to_file(out);
        ms_weights_hg.to_file(out);
        ms_weights_hi.to_file(out);
        ms_weights_hf.to_file(out);
        ms_weights_ho.to_file(out);
        ms_bias_g.to_file(out);
        ms_bias_i.to_file(out);
        ms_bias_f.to_file(out);
        ms_bias_o.to_file(out);
    }

    inline void from_file(std::ifstream &in) noexcept
    {
        weights_xg.from_file(in);
        weights_xi.from_file(in);
        weights_xf.from_file(in);
        weights_xo.from_file(in);
        weights_hg.from_file(in);
        weights_hi.from_file(in);
        weights_hf.from_file(in);
        weights_ho.from_file(in);
        bias_g.from_file(in);
        bias_i.from_file(in);
        bias_f.from_file(in);
        bias_o.from_file(in);

        moment_weights_xg.from_file(in);
        moment_weights_xi.from_file(in);
        moment_weights_xf.from_file(in);
        moment_weights_xo.from_file(in);
        moment_weights_hg.from_file(in);
        moment_weights_hi.from_file(in);
        moment_weights_hf.from_file(in);
        moment_weights_ho.from_file(in);
        moment_bias_g.from_file(in);
        moment_bias_i.from_file(in);
        moment_bias_f.from_file(in);
        moment_bias_o.from_file(in);

        ms_weights_xg.from_file(in);
        ms_weights_xi.from_file(in);
        ms_weights_xf.from_file(in);
        ms_weights_xo.from_file(in);
        ms_weights_hg.from_file(in);
        ms_weights_hi.from_file(in);
        ms_weights_hf.from_file(in);
        ms_weights_ho.from_file(in);
        ms_bias_g.from_file(in);
        ms_bias_i.from_file(in);
        ms_bias_f.from_file(in);
        ms_bias_o.from_file(in);
    }

    inline void to_bin_file(std::ofstream &out) noexcept
    {
        weights_xg.to_bin_file(out);
        weights_xi.to_bin_file(out);
        weights_xf.to_bin_file(out);
        weights_xo.to_bin_file(out);
        weights_hg.to_bin_file(out);
        weights_hi.to_bin_file(out);
        weights_hf.to_bin_file(out);
        weights_ho.to_bin_file(out);
        bias_g.to_bin_file(out);
        bias_i.to_bin_file(out);
        bias_f.to_bin_file(out);
        bias_o.to_bin_file(out);

        moment_weights_xg.to_bin_file(out);
        moment_weights_xi.to_bin_file(out);
        moment_weights_xf.to_bin_file(out);
        moment_weights_xo.to_bin_file(out);
        moment_weights_hg.to_bin_file(out);
        moment_weights_hi.to_bin_file(out);
        moment_weights_hf.to_bin_file(out);
        moment_weights_ho.to_bin_file(out);
        moment_bias_g.to_bin_file(out);
        moment_bias_i.to_bin_file(out);
        moment_bias_f.to_bin_file(out);
        moment_bias_o.to_bin_file(out);

        ms_weights_xg.to_bin_file(out);
        ms_weights_xi.to_bin_file(out);
        ms_weights_xf.to_bin_file(out);
        ms_weights_xo.to_bin_file(out);
        ms_weights_hg.to_bin_file(out);
        ms_weights_hi.to_bin_file(out);
        ms_weights_hf.to_bin_file(out);
        ms_weights_ho.to_bin_file(out);
        ms_bias_g.to_bin_file(out);
        ms_bias_i.to_bin_file(out);
        ms_bias_f.to_bin_file(out);
        ms_bias_o.to_bin_file(out);
    }

    inline void from_bin_file(std::ifstream &in) noexcept
    {
        weights_xg.from_bin_file(in);
        weights_xi.from_bin_file(in);
        weights_xf.from_bin_file(in);
        weights_xo.from_bin_file(in);
        weights_hg.from_bin_file(in);
        weights_hi.from_bin_file(in);
        weights_hf.from_bin_file(in);
        weights_ho.from_bin_file(in);
        bias_g.from_bin_file(in);
        bias_i.from_bin_file(in);
        bias_f.from_bin_file(in);
        bias_o.from_bin_file(in);

        moment_weights_xg.from_bin_file(in);
        moment_weights_xi.from_bin_file(in);
        moment_weights_xf.from_bin_file(in);
        moment_weights_xo.from_bin_file(in);
        moment_weights_hg.from_bin_file(in);
        moment_weights_hi.from_bin_file(in);
        moment_weights_hf.from_bin_file(in);
        moment_weights_ho.from_bin_file(in);
        moment_bias_g.from_bin_file(in);
        moment_bias_i.from_bin_file(in);
        moment_bias_f.from_bin_file(in);
        moment_bias_o.from_bin_file(in);

        ms_weights_xg.from_bin_file(in);
        ms_weights_xi.from_bin_file(in);
        ms_weights_xf.from_bin_file(in);
        ms_weights_xo.from_bin_file(in);
        ms_weights_hg.from_bin_file(in);
        ms_weights_hi.from_bin_file(in);
        ms_weights_hf.from_bin_file(in);
        ms_weights_ho.from_bin_file(in);
        ms_bias_g.from_bin_file(in);
        ms_bias_i.from_bin_file(in);
        ms_bias_f.from_bin_file(in);
        ms_bias_o.from_bin_file(in);
    }

    inline void apply_momentum(const double momentum) noexcept
    {
        moment_weights_xg.mul(momentum);
        moment_weights_xi.mul(momentum);
        moment_weights_xf.mul(momentum);
        moment_weights_xo.mul(momentum);
        moment_weights_hg.mul(momentum);
        moment_weights_hi.mul(momentum);
        moment_weights_hf.mul(momentum);
        moment_weights_ho.mul(momentum);
        moment_bias_g.mul(momentum);
        moment_bias_i.mul(momentum);
        moment_bias_f.mul(momentum);
        moment_bias_o.mul(momentum);
        weights_xg.add(moment_weights_xg);
        weights_xi.add(moment_weights_xi);
        weights_xf.add(moment_weights_xf);
        weights_xo.add(moment_weights_xo);
        weights_hg.add(moment_weights_hg);
        weights_hi.add(moment_weights_hi);
        weights_hf.add(moment_weights_hf);
        weights_ho.add(moment_weights_ho);
        bias_g.add(moment_bias_g);
        bias_i.add(moment_bias_i);
        bias_f.add(moment_bias_f);
        bias_o.add(moment_bias_o);
    }

    inline void update_weights_momentum_ms(const double learning_rate, const double decay) noexcept
    {
        ms_weights_xg.mul(decay);
        ms_weights_xi.mul(decay);
        ms_weights_xf.mul(decay);
        ms_weights_xo.mul(decay);
        ms_weights_hg.mul(decay);
        ms_weights_hi.mul(decay);
        ms_weights_hf.mul(decay);
        ms_weights_ho.mul(decay);
        ms_bias_g.mul(decay);
        ms_bias_i.mul(decay);
        ms_bias_f.mul(decay);
        ms_bias_o.mul(decay);

        ms_weights_xg.add_factor_mul_a_squared(1-decay, weights_xg_gradient_acc);
        ms_weights_xi.add_factor_mul_a_squared(1-decay, weights_xi_gradient_acc);
        ms_weights_xf.add_factor_mul_a_squared(1-decay, weights_xf_gradient_acc);
        ms_weights_xo.add_factor_mul_a_squared(1-decay, weights_xo_gradient_acc);
        ms_weights_hg.add_factor_mul_a_squared(1-decay, weights_hg_gradient_acc);
        ms_weights_hi.add_factor_mul_a_squared(1-decay, weights_hi_gradient_acc);
        ms_weights_hf.add_factor_mul_a_squared(1-decay, weights_hf_gradient_acc);
        ms_weights_ho.add_factor_mul_a_squared(1-decay, weights_ho_gradient_acc);
        ms_bias_g.add_factor_mul_a_squared(1-decay, bias_g_gradient_acc);
        ms_bias_i.add_factor_mul_a_squared(1-decay, bias_i_gradient_acc);
        ms_bias_f.add_factor_mul_a_squared(1-decay, bias_f_gradient_acc);
        ms_bias_o.add_factor_mul_a_squared(1-decay, bias_o_gradient_acc);

        update_weight_momentum(weights_xg, moment_weights_xg, ms_weights_xg, weights_xg_gradient_acc, learning_rate);
        update_weight_momentum(weights_xi, moment_weights_xi, ms_weights_xi, weights_xi_gradient_acc, learning_rate);
        update_weight_momentum(weights_xf, moment_weights_xf, ms_weights_xf, weights_xf_gradient_acc, learning_rate);
        update_weight_momentum(weights_xo, moment_weights_xo, ms_weights_xo, weights_xo_gradient_acc, learning_rate);
        update_weight_momentum(weights_hg, moment_weights_hg, ms_weights_hg, weights_hg_gradient_acc, learning_rate);
        update_weight_momentum(weights_hi, moment_weights_hi, ms_weights_hi, weights_hi_gradient_acc, learning_rate);
        update_weight_momentum(weights_hf, moment_weights_hf, ms_weights_hf, weights_hf_gradient_acc, learning_rate);
        update_weight_momentum(weights_ho, moment_weights_ho, ms_weights_ho, weights_ho_gradient_acc, learning_rate);
        update_weight_momentum(bias_g, moment_bias_g, ms_bias_g, bias_g_gradient_acc, learning_rate);
        update_weight_momentum(bias_i, moment_bias_i, ms_bias_i, bias_i_gradient_acc, learning_rate);
        update_weight_momentum(bias_f, moment_bias_f, ms_bias_f, bias_f_gradient_acc, learning_rate);
        update_weight_momentum(bias_o, moment_bias_o, ms_bias_o, bias_o_gradient_acc, learning_rate);

        weights_xg_gradient_acc.set(0.0);
        weights_xi_gradient_acc.set(0.0);
        weights_xf_gradient_acc.set(0.0);
        weights_xo_gradient_acc.set(0.0);
        weights_hg_gradient_acc.set(0.0);
        weights_hi_gradient_acc.set(0.0);
        weights_hf_gradient_acc.set(0.0);
        weights_ho_gradient_acc.set(0.0);
        bias_g_gradient_acc.set(0.0);
        bias_i_gradient_acc.set(0.0);
        bias_f_gradient_acc.set(0.0);
        bias_o_gradient_acc.set(0.0);
    }
};


template<unsigned long input_size, unsigned long mem_cell_size>
class RMSPropLSTMBlock : public BaseLSTMBlock<input_size, mem_cell_size>
{
private:
    //LSTM ms for weights
    Matrix<input_size, mem_cell_size> ms_weights_xg;
    Matrix<input_size, mem_cell_size> ms_weights_xi;
    Matrix<input_size, mem_cell_size> ms_weights_xf;
    Matrix<input_size, mem_cell_size> ms_weights_xo;
    Matrix<mem_cell_size, mem_cell_size> ms_weights_hg;
    Matrix<mem_cell_size, mem_cell_size> ms_weights_hi;
    Matrix<mem_cell_size, mem_cell_size> ms_weights_hf;
    Matrix<mem_cell_size, mem_cell_size> ms_weights_ho;
    //LSTM ms for biases
    Matrix<1, mem_cell_size> ms_bias_g;
    Matrix<1, mem_cell_size> ms_bias_i;
    Matrix<1, mem_cell_size> ms_bias_f;
    Matrix<1, mem_cell_size> ms_bias_o;
public:
    using BaseLSTMBlock<input_size,mem_cell_size>::lstm_states;

    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xg;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xi;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xf;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xo;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hg;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hi;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hf;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_ho;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_g;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_i;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_f;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_o;

    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xg_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xi_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xf_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xo_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hg_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hi_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hf_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_ho_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_g_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_i_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_f_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_o_gradient_acc;
    RMSPropLSTMBlock(size_t time_steps=0) noexcept:BaseLSTMBlock<input_size,mem_cell_size>(time_steps)
    , ms_weights_xg(1.0), ms_weights_xi(1.0), ms_weights_xf(1.0), ms_weights_xo(1.0)
    , ms_weights_hg(1.0), ms_weights_hi(1.0), ms_weights_hf(1.0), ms_weights_ho(1.0)
    , ms_bias_g(1.0), ms_bias_i(1.0), ms_bias_f(1.0), ms_bias_o(1.0)
    {
    }

    inline void to_file(std::ofstream &out) noexcept
    {
        weights_xg.to_file(out);
        weights_xi.to_file(out);
        weights_xf.to_file(out);
        weights_xo.to_file(out);
        weights_hg.to_file(out);
        weights_hi.to_file(out);
        weights_hf.to_file(out);
        weights_ho.to_file(out);
        bias_g.to_file(out);
        bias_i.to_file(out);
        bias_f.to_file(out);
        bias_o.to_file(out);

        ms_weights_xg.to_file(out);
        ms_weights_xi.to_file(out);
        ms_weights_xf.to_file(out);
        ms_weights_xo.to_file(out);
        ms_weights_hg.to_file(out);
        ms_weights_hi.to_file(out);
        ms_weights_hf.to_file(out);
        ms_weights_ho.to_file(out);
        ms_bias_g.to_file(out);
        ms_bias_i.to_file(out);
        ms_bias_f.to_file(out);
        ms_bias_o.to_file(out);
    }

    inline void from_file(std::ifstream &in) noexcept
    {
        weights_xg.from_file(in);
        weights_xi.from_file(in);
        weights_xf.from_file(in);
        weights_xo.from_file(in);
        weights_hg.from_file(in);
        weights_hi.from_file(in);
        weights_hf.from_file(in);
        weights_ho.from_file(in);
        bias_g.from_file(in);
        bias_i.from_file(in);
        bias_f.from_file(in);
        bias_o.from_file(in);

        ms_weights_xg.from_file(in);
        ms_weights_xi.from_file(in);
        ms_weights_xf.from_file(in);
        ms_weights_xo.from_file(in);
        ms_weights_hg.from_file(in);
        ms_weights_hi.from_file(in);
        ms_weights_hf.from_file(in);
        ms_weights_ho.from_file(in);
        ms_bias_g.from_file(in);
        ms_bias_i.from_file(in);
        ms_bias_f.from_file(in);
        ms_bias_o.from_file(in);
    }

    inline void to_bin_file(std::ofstream &out) noexcept
    {
        weights_xg.to_bin_file(out);
        weights_xi.to_bin_file(out);
        weights_xf.to_bin_file(out);
        weights_xo.to_bin_file(out);
        weights_hg.to_bin_file(out);
        weights_hi.to_bin_file(out);
        weights_hf.to_bin_file(out);
        weights_ho.to_bin_file(out);
        bias_g.to_bin_file(out);
        bias_i.to_bin_file(out);
        bias_f.to_bin_file(out);
        bias_o.to_bin_file(out);

        ms_weights_xg.to_bin_file(out);
        ms_weights_xi.to_bin_file(out);
        ms_weights_xf.to_bin_file(out);
        ms_weights_xo.to_bin_file(out);
        ms_weights_hg.to_bin_file(out);
        ms_weights_hi.to_bin_file(out);
        ms_weights_hf.to_bin_file(out);
        ms_weights_ho.to_bin_file(out);
        ms_bias_g.to_bin_file(out);
        ms_bias_i.to_bin_file(out);
        ms_bias_f.to_bin_file(out);
        ms_bias_o.to_bin_file(out);
    }

    inline void from_bin_file(std::ifstream &in) noexcept
    {
        weights_xg.from_bin_file(in);
        weights_xi.from_bin_file(in);
        weights_xf.from_bin_file(in);
        weights_xo.from_bin_file(in);
        weights_hg.from_bin_file(in);
        weights_hi.from_bin_file(in);
        weights_hf.from_bin_file(in);
        weights_ho.from_bin_file(in);
        bias_g.from_bin_file(in);
        bias_i.from_bin_file(in);
        bias_f.from_bin_file(in);
        bias_o.from_bin_file(in);

        ms_weights_xg.from_bin_file(in);
        ms_weights_xi.from_bin_file(in);
        ms_weights_xf.from_bin_file(in);
        ms_weights_xo.from_bin_file(in);
        ms_weights_hg.from_bin_file(in);
        ms_weights_hi.from_bin_file(in);
        ms_weights_hf.from_bin_file(in);
        ms_weights_ho.from_bin_file(in);
        ms_bias_g.from_bin_file(in);
        ms_bias_i.from_bin_file(in);
        ms_bias_f.from_bin_file(in);
        ms_bias_o.from_bin_file(in);
    }

    inline void update_weights_ms(const double learning_rate, const double decay) noexcept
    {
        ms_weights_xg.mul(decay);
        ms_weights_xi.mul(decay);
        ms_weights_xf.mul(decay);
        ms_weights_xo.mul(decay);
        ms_weights_hg.mul(decay);
        ms_weights_hi.mul(decay);
        ms_weights_hf.mul(decay);
        ms_weights_ho.mul(decay);
        ms_bias_g.mul(decay);
        ms_bias_i.mul(decay);
        ms_bias_f.mul(decay);
        ms_bias_o.mul(decay);

        ms_weights_xg.add_factor_mul_a_squared(1-decay, weights_xg_gradient_acc);
        ms_weights_xi.add_factor_mul_a_squared(1-decay, weights_xi_gradient_acc);
        ms_weights_xf.add_factor_mul_a_squared(1-decay, weights_xf_gradient_acc);
        ms_weights_xo.add_factor_mul_a_squared(1-decay, weights_xo_gradient_acc);
        ms_weights_hg.add_factor_mul_a_squared(1-decay, weights_hg_gradient_acc);
        ms_weights_hi.add_factor_mul_a_squared(1-decay, weights_hi_gradient_acc);
        ms_weights_hf.add_factor_mul_a_squared(1-decay, weights_hf_gradient_acc);
        ms_weights_ho.add_factor_mul_a_squared(1-decay, weights_ho_gradient_acc);
        ms_bias_g.add_factor_mul_a_squared(1-decay, bias_g_gradient_acc);
        ms_bias_i.add_factor_mul_a_squared(1-decay, bias_i_gradient_acc);
        ms_bias_f.add_factor_mul_a_squared(1-decay, bias_f_gradient_acc);
        ms_bias_o.add_factor_mul_a_squared(1-decay, bias_o_gradient_acc);

        update_weight_with_ms(weights_xg, ms_weights_xg, weights_xg_gradient_acc, learning_rate);
        update_weight_with_ms(weights_xi, ms_weights_xi, weights_xi_gradient_acc, learning_rate);
        update_weight_with_ms(weights_xf, ms_weights_xf, weights_xf_gradient_acc, learning_rate);
        update_weight_with_ms(weights_xo, ms_weights_xo, weights_xo_gradient_acc, learning_rate);
        update_weight_with_ms(weights_hg, ms_weights_hg, weights_hg_gradient_acc, learning_rate);
        update_weight_with_ms(weights_hi, ms_weights_hi, weights_hi_gradient_acc, learning_rate);
        update_weight_with_ms(weights_hf, ms_weights_hf, weights_hf_gradient_acc, learning_rate);
        update_weight_with_ms(weights_ho, ms_weights_ho, weights_ho_gradient_acc, learning_rate);
        update_weight_with_ms(bias_g, ms_bias_g, bias_g_gradient_acc, learning_rate);
        update_weight_with_ms(bias_i, ms_bias_i, bias_i_gradient_acc, learning_rate);
        update_weight_with_ms(bias_f, ms_bias_f, bias_f_gradient_acc, learning_rate);
        update_weight_with_ms(bias_o, ms_bias_o, bias_o_gradient_acc, learning_rate);

        weights_xg_gradient_acc.set(0.0);
        weights_xi_gradient_acc.set(0.0);
        weights_xf_gradient_acc.set(0.0);
        weights_xo_gradient_acc.set(0.0);
        weights_hg_gradient_acc.set(0.0);
        weights_hi_gradient_acc.set(0.0);
        weights_hf_gradient_acc.set(0.0);
        weights_ho_gradient_acc.set(0.0);
        bias_g_gradient_acc.set(0.0);
        bias_i_gradient_acc.set(0.0);
        bias_f_gradient_acc.set(0.0);
        bias_o_gradient_acc.set(0.0);
    }
};

template<unsigned long input_size, unsigned long mem_cell_size>
class AdamLSTMBlock : public BaseLSTMBlock<input_size, mem_cell_size>
{
private:
    //LSTM ms for weights
    Matrix<input_size, mem_cell_size> ms_weights_xg;
    Matrix<input_size, mem_cell_size> ms_weights_xi;
    Matrix<input_size, mem_cell_size> ms_weights_xf;
    Matrix<input_size, mem_cell_size> ms_weights_xo;
    Matrix<mem_cell_size, mem_cell_size> ms_weights_hg;
    Matrix<mem_cell_size, mem_cell_size> ms_weights_hi;
    Matrix<mem_cell_size, mem_cell_size> ms_weights_hf;
    Matrix<mem_cell_size, mem_cell_size> ms_weights_ho;
    //LSTM ms for biases
    Matrix<1, mem_cell_size> ms_bias_g;
    Matrix<1, mem_cell_size> ms_bias_i;
    Matrix<1, mem_cell_size> ms_bias_f;
    Matrix<1, mem_cell_size> ms_bias_o;

    //LSTM mns for weights
    Matrix<input_size, mem_cell_size> mns_weights_xg;
    Matrix<input_size, mem_cell_size> mns_weights_xi;
    Matrix<input_size, mem_cell_size> mns_weights_xf;
    Matrix<input_size, mem_cell_size> mns_weights_xo;
    Matrix<mem_cell_size, mem_cell_size> mns_weights_hg;
    Matrix<mem_cell_size, mem_cell_size> mns_weights_hi;
    Matrix<mem_cell_size, mem_cell_size> mns_weights_hf;
    Matrix<mem_cell_size, mem_cell_size> mns_weights_ho;
    //LSTM mns for biases
    Matrix<1, mem_cell_size> mns_bias_g;
    Matrix<1, mem_cell_size> mns_bias_i;
    Matrix<1, mem_cell_size> mns_bias_f;
    Matrix<1, mem_cell_size> mns_bias_o;
public:
    using BaseLSTMBlock<input_size,mem_cell_size>::lstm_states;

    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xg;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xi;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xf;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xo;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hg;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hi;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hf;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_ho;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_g;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_i;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_f;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_o;

    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xg_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xi_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xf_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_xo_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hg_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hi_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_hf_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::weights_ho_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_g_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_i_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_f_gradient_acc;
    using BaseLSTMBlock<input_size,mem_cell_size>::bias_o_gradient_acc;
    AdamLSTMBlock(size_t time_steps=0) noexcept:BaseLSTMBlock<input_size,mem_cell_size>(time_steps)
    , ms_weights_xg(1.0), ms_weights_xi(1.0), ms_weights_xf(1.0), ms_weights_xo(1.0)
    , ms_weights_hg(1.0), ms_weights_hi(1.0), ms_weights_hf(1.0), ms_weights_ho(1.0)
    , ms_bias_g(1.0), ms_bias_i(1.0), ms_bias_f(1.0), ms_bias_o(1.0)
    , mns_weights_xg(1.0), mns_weights_xi(1.0), mns_weights_xf(1.0), mns_weights_xo(1.0)
    , mns_weights_hg(1.0), mns_weights_hi(1.0), mns_weights_hf(1.0), mns_weights_ho(1.0)
    , mns_bias_g(1.0), mns_bias_i(1.0), mns_bias_f(1.0), mns_bias_o(1.0)
    {
    }

    inline void to_file(std::ofstream &out) noexcept
    {
        weights_xg.to_file(out);
        weights_xi.to_file(out);
        weights_xf.to_file(out);
        weights_xo.to_file(out);
        weights_hg.to_file(out);
        weights_hi.to_file(out);
        weights_hf.to_file(out);
        weights_ho.to_file(out);
        bias_g.to_file(out);
        bias_i.to_file(out);
        bias_f.to_file(out);
        bias_o.to_file(out);

        ms_weights_xg.to_file(out);
        ms_weights_xi.to_file(out);
        ms_weights_xf.to_file(out);
        ms_weights_xo.to_file(out);
        ms_weights_hg.to_file(out);
        ms_weights_hi.to_file(out);
        ms_weights_hf.to_file(out);
        ms_weights_ho.to_file(out);
        ms_bias_g.to_file(out);
        ms_bias_i.to_file(out);
        ms_bias_f.to_file(out);
        ms_bias_o.to_file(out);

        mns_weights_xg.to_file(out);
        mns_weights_xi.to_file(out);
        mns_weights_xf.to_file(out);
        mns_weights_xo.to_file(out);
        mns_weights_hg.to_file(out);
        mns_weights_hi.to_file(out);
        mns_weights_hf.to_file(out);
        mns_weights_ho.to_file(out);
        mns_bias_g.to_file(out);
        mns_bias_i.to_file(out);
        mns_bias_f.to_file(out);
        mns_bias_o.to_file(out);
    }

    inline void from_file(std::ifstream &in) noexcept
    {
        weights_xg.from_file(in);
        weights_xi.from_file(in);
        weights_xf.from_file(in);
        weights_xo.from_file(in);
        weights_hg.from_file(in);
        weights_hi.from_file(in);
        weights_hf.from_file(in);
        weights_ho.from_file(in);
        bias_g.from_file(in);
        bias_i.from_file(in);
        bias_f.from_file(in);
        bias_o.from_file(in);

        ms_weights_xg.from_file(in);
        ms_weights_xi.from_file(in);
        ms_weights_xf.from_file(in);
        ms_weights_xo.from_file(in);
        ms_weights_hg.from_file(in);
        ms_weights_hi.from_file(in);
        ms_weights_hf.from_file(in);
        ms_weights_ho.from_file(in);
        ms_bias_g.from_file(in);
        ms_bias_i.from_file(in);
        ms_bias_f.from_file(in);
        ms_bias_o.from_file(in);

        mns_weights_xg.from_file(in);
        mns_weights_xi.from_file(in);
        mns_weights_xf.from_file(in);
        mns_weights_xo.from_file(in);
        mns_weights_hg.from_file(in);
        mns_weights_hi.from_file(in);
        mns_weights_hf.from_file(in);
        mns_weights_ho.from_file(in);
        mns_bias_g.from_file(in);
        mns_bias_i.from_file(in);
        mns_bias_f.from_file(in);
        mns_bias_o.from_file(in);
    }

    inline void to_bin_file(std::ofstream &out) noexcept
    {
        weights_xg.to_bin_file(out);
        weights_xi.to_bin_file(out);
        weights_xf.to_bin_file(out);
        weights_xo.to_bin_file(out);
        weights_hg.to_bin_file(out);
        weights_hi.to_bin_file(out);
        weights_hf.to_bin_file(out);
        weights_ho.to_bin_file(out);
        bias_g.to_bin_file(out);
        bias_i.to_bin_file(out);
        bias_f.to_bin_file(out);
        bias_o.to_bin_file(out);

        ms_weights_xg.to_bin_file(out);
        ms_weights_xi.to_bin_file(out);
        ms_weights_xf.to_bin_file(out);
        ms_weights_xo.to_bin_file(out);
        ms_weights_hg.to_bin_file(out);
        ms_weights_hi.to_bin_file(out);
        ms_weights_hf.to_bin_file(out);
        ms_weights_ho.to_bin_file(out);
        ms_bias_g.to_bin_file(out);
        ms_bias_i.to_bin_file(out);
        ms_bias_f.to_bin_file(out);
        ms_bias_o.to_bin_file(out);

        mns_weights_xg.to_bin_file(out);
        mns_weights_xi.to_bin_file(out);
        mns_weights_xf.to_bin_file(out);
        mns_weights_xo.to_bin_file(out);
        mns_weights_hg.to_bin_file(out);
        mns_weights_hi.to_bin_file(out);
        mns_weights_hf.to_bin_file(out);
        mns_weights_ho.to_bin_file(out);
        mns_bias_g.to_bin_file(out);
        mns_bias_i.to_bin_file(out);
        mns_bias_f.to_bin_file(out);
        mns_bias_o.to_bin_file(out);
    }

    inline void from_bin_file(std::ifstream &in) noexcept
    {
        weights_xg.from_bin_file(in);
        weights_xi.from_bin_file(in);
        weights_xf.from_bin_file(in);
        weights_xo.from_bin_file(in);
        weights_hg.from_bin_file(in);
        weights_hi.from_bin_file(in);
        weights_hf.from_bin_file(in);
        weights_ho.from_bin_file(in);
        bias_g.from_bin_file(in);
        bias_i.from_bin_file(in);
        bias_f.from_bin_file(in);
        bias_o.from_bin_file(in);

        ms_weights_xg.from_bin_file(in);
        ms_weights_xi.from_bin_file(in);
        ms_weights_xf.from_bin_file(in);
        ms_weights_xo.from_bin_file(in);
        ms_weights_hg.from_bin_file(in);
        ms_weights_hi.from_bin_file(in);
        ms_weights_hf.from_bin_file(in);
        ms_weights_ho.from_bin_file(in);
        ms_bias_g.from_bin_file(in);
        ms_bias_i.from_bin_file(in);
        ms_bias_f.from_bin_file(in);
        ms_bias_o.from_bin_file(in);

        mns_weights_xg.from_bin_file(in);
        mns_weights_xi.from_bin_file(in);
        mns_weights_xf.from_bin_file(in);
        mns_weights_xo.from_bin_file(in);
        mns_weights_hg.from_bin_file(in);
        mns_weights_hi.from_bin_file(in);
        mns_weights_hf.from_bin_file(in);
        mns_weights_ho.from_bin_file(in);
        mns_bias_g.from_bin_file(in);
        mns_bias_i.from_bin_file(in);
        mns_bias_f.from_bin_file(in);
        mns_bias_o.from_bin_file(in);
    }

    inline void update_weights_adam(const double learning_rate, const double decay1, const double decay2) noexcept
    {
        ms_weights_xg.mul(decay1);
        ms_weights_xi.mul(decay1);
        ms_weights_xf.mul(decay1);
        ms_weights_xo.mul(decay1);
        ms_weights_hg.mul(decay1);
        ms_weights_hi.mul(decay1);
        ms_weights_hf.mul(decay1);
        ms_weights_ho.mul(decay1);
        ms_bias_g.mul(decay1);
        ms_bias_i.mul(decay1);
        ms_bias_f.mul(decay1);
        ms_bias_o.mul(decay1);

        mns_weights_xg.mul(decay2);
        mns_weights_xi.mul(decay2);
        mns_weights_xf.mul(decay2);
        mns_weights_xo.mul(decay2);
        mns_weights_hg.mul(decay2);
        mns_weights_hi.mul(decay2);
        mns_weights_hf.mul(decay2);
        mns_weights_ho.mul(decay2);
        mns_bias_g.mul(decay2);
        mns_bias_i.mul(decay2);
        mns_bias_f.mul(decay2);
        mns_bias_o.mul(decay2);

        ms_weights_xg.add_factor_mul_a_squared(1-decay1, weights_xg_gradient_acc);
        ms_weights_xi.add_factor_mul_a_squared(1-decay1, weights_xi_gradient_acc);
        ms_weights_xf.add_factor_mul_a_squared(1-decay1, weights_xf_gradient_acc);
        ms_weights_xo.add_factor_mul_a_squared(1-decay1, weights_xo_gradient_acc);
        ms_weights_hg.add_factor_mul_a_squared(1-decay1, weights_hg_gradient_acc);
        ms_weights_hi.add_factor_mul_a_squared(1-decay1, weights_hi_gradient_acc);
        ms_weights_hf.add_factor_mul_a_squared(1-decay1, weights_hf_gradient_acc);
        ms_weights_ho.add_factor_mul_a_squared(1-decay1, weights_ho_gradient_acc);
        ms_bias_g.add_factor_mul_a_squared(1-decay1, bias_g_gradient_acc);
        ms_bias_i.add_factor_mul_a_squared(1-decay1, bias_i_gradient_acc);
        ms_bias_f.add_factor_mul_a_squared(1-decay1, bias_f_gradient_acc);
        ms_bias_o.add_factor_mul_a_squared(1-decay1, bias_o_gradient_acc);

        mns_weights_xg.add_factor_mul_a(1-decay2, weights_xg_gradient_acc);
        mns_weights_xi.add_factor_mul_a(1-decay2, weights_xi_gradient_acc);
        mns_weights_xf.add_factor_mul_a(1-decay2, weights_xf_gradient_acc);
        mns_weights_xo.add_factor_mul_a(1-decay2, weights_xo_gradient_acc);
        mns_weights_hg.add_factor_mul_a(1-decay2, weights_hg_gradient_acc);
        mns_weights_hi.add_factor_mul_a(1-decay2, weights_hi_gradient_acc);
        mns_weights_hf.add_factor_mul_a(1-decay2, weights_hf_gradient_acc);
        mns_weights_ho.add_factor_mul_a(1-decay2, weights_ho_gradient_acc);
        mns_bias_g.add_factor_mul_a(1-decay2, bias_g_gradient_acc);
        mns_bias_i.add_factor_mul_a(1-decay2, bias_i_gradient_acc);
        mns_bias_f.add_factor_mul_a(1-decay2, bias_f_gradient_acc);
        mns_bias_o.add_factor_mul_a(1-decay2, bias_o_gradient_acc);

        update_weight_with_adam(weights_xg, ms_weights_xg, mns_weights_xg, learning_rate, decay1, decay2);
        update_weight_with_adam(weights_xi, ms_weights_xi, mns_weights_xi, learning_rate, decay1, decay2);
        update_weight_with_adam(weights_xf, ms_weights_xf, mns_weights_xf, learning_rate, decay1, decay2);
        update_weight_with_adam(weights_xo, ms_weights_xo, mns_weights_xo, learning_rate, decay1, decay2);
        update_weight_with_adam(weights_hg, ms_weights_hg, mns_weights_hg, learning_rate, decay1, decay2);
        update_weight_with_adam(weights_hi, ms_weights_hi, mns_weights_hi, learning_rate, decay1, decay2);
        update_weight_with_adam(weights_hf, ms_weights_hf, mns_weights_hf, learning_rate, decay1, decay2);
        update_weight_with_adam(weights_ho, ms_weights_ho, mns_weights_ho, learning_rate, decay1, decay2);
        update_weight_with_adam(bias_g, ms_bias_g, mns_bias_g, learning_rate, decay1, decay2);
        update_weight_with_adam(bias_i, ms_bias_i, mns_bias_i, learning_rate, decay1, decay2);
        update_weight_with_adam(bias_f, ms_bias_f, mns_bias_f, learning_rate, decay1, decay2);
        update_weight_with_adam(bias_o, ms_bias_o, mns_bias_o, learning_rate, decay1, decay2);

        weights_xg_gradient_acc.set(0.0);
        weights_xi_gradient_acc.set(0.0);
        weights_xf_gradient_acc.set(0.0);
        weights_xo_gradient_acc.set(0.0);
        weights_hg_gradient_acc.set(0.0);
        weights_hi_gradient_acc.set(0.0);
        weights_hf_gradient_acc.set(0.0);
        weights_ho_gradient_acc.set(0.0);
        bias_g_gradient_acc.set(0.0);
        bias_i_gradient_acc.set(0.0);
        bias_f_gradient_acc.set(0.0);
        bias_o_gradient_acc.set(0.0);
    }
};