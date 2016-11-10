#include "matrix.hpp"
#include <vector>

#ifndef __LAYERSTATESTRUCT__
#define __LAYERSTATESTRUCT__
template<unsigned long input_size, unsigned long output_size>
struct LayerState
{
    Matrix<1,output_size> output;
    Matrix<1,output_size> delta_output;
};
#endif

template<unsigned long input_size, unsigned long output_size>
class BaseSoftmaxBlock
{
protected:
    std::vector<LayerState<input_size, output_size>> layer_states;
    Matrix<input_size, output_size> weights;
    Matrix<1, output_size> bias;
    Matrix<input_size, output_size> weight_gradient_accumulator;
    Matrix<1, output_size> bias_gradient_accumulator;
public:
    BaseSoftmaxBlock(size_t time_steps=0) noexcept:layer_states(time_steps), weight_gradient_accumulator(0.0), bias_gradient_accumulator(0.0)
    {
        weights.randomize_for_nn(input_size+1);
        bias.randomize_for_nn(input_size+1);
    }

    inline void only_wb_to_bin_file(std::ofstream &out) const noexcept
    {
        weights.to_bin_file(out);
        bias.to_bin_file(out);
    }

    inline void only_wb_from_bin_file(std::ifstream &in) noexcept
    {
        weights.from_bin_file(in);
        bias.from_bin_file(in);
    }

    inline void set_time_steps(size_t time_steps) noexcept
    {
        layer_states.resize(time_steps);
    }

    inline void reserve_time_steps(size_t time_steps) noexcept
    {
        layer_states.reserve(time_steps);
    }

    inline void calc(const Matrix<1,input_size> &X, size_t time_step) noexcept
    {
        assert(time_step<layer_states.size());

        layer_states[time_step].output.equals_a_dot_b(X, weights);
        layer_states[time_step].output.add(bias);
        layer_states[time_step].output.apply_softmax();
    }

    inline void set_first_delta_and_propagate_with_cross_enthropy(const Matrix<1,output_size> &Y, Matrix<1,input_size> &X_delta, size_t time_step) noexcept
    {
        assert(time_step<layer_states.size());
        //Get outputs delta
        layer_states[time_step].delta_output.equals_a_sub_b(Y,layer_states[time_step].output);
        X_delta.equals_a_dot_bt(layer_states[time_step].delta_output, weights);
    }

    inline void set_first_delta(const Matrix<1,output_size> &Y, size_t time_step) noexcept
    {
        assert(time_step<layer_states.size());
        //Get outputs delta
        layer_states[time_step].delta_output.equals_a_sub_b(Y,layer_states[time_step].output);
    }

    inline void propagate_delta(size_t time_step) noexcept
    {
        assert(time_step<layer_states.size());
        //Propagate delta
        layer_states[time_step].delta_output.mult_after_func03(layer_states[time_step].output);
    }

    inline void propagate_delta(Matrix<1,input_size> &X_delta, size_t time_step) noexcept
    {
        propagate_delta(time_step);
        //Propagate delta to next building block
        X_delta.equals_a_dot_bt(layer_states[time_step].delta_output, weights);
    }

    inline void accumulate_gradients(const Matrix<1,input_size> &X, size_t time_step) noexcept
    {
        weight_gradient_accumulator.add_at_dot_b(X, layer_states[time_step].delta_output);
        bias_gradient_accumulator.add(layer_states[time_step].delta_output);
    }

    inline const Matrix<1,output_size>& get_output(size_t time_step) const noexcept
    {
        return layer_states[time_step].output;
    }

    inline Matrix<1,output_size>& get_delta_output(size_t time_step) noexcept
    {
        return layer_states[time_step].delta_output;
    }
};

template<unsigned long input_size, unsigned long output_size>
class NAGSoftmaxBlock : public BaseSoftmaxBlock<input_size, output_size>
{
private:
    Matrix<input_size, output_size> moment_weights;
    Matrix<1, output_size> moment_bias;
public:
    using BaseSoftmaxBlock<input_size, output_size>::layer_states;
    using BaseSoftmaxBlock<input_size, output_size>::weights;
    using BaseSoftmaxBlock<input_size, output_size>::bias;
    using BaseSoftmaxBlock<input_size, output_size>::weight_gradient_accumulator;
    using BaseSoftmaxBlock<input_size, output_size>::bias_gradient_accumulator;
    NAGSoftmaxBlock(size_t time_steps=0) noexcept
    :BaseSoftmaxBlock<input_size, output_size>(time_steps),moment_weights(0.0),moment_bias(0.0)
    {
    }

    inline void reset_momentum() noexcept
    {
        moment_weights.set(0.0);
        moment_bias.set(0.0);
    }

    inline void to_file(std::ofstream &out) const noexcept
    {
        weights.to_file(out);
        bias.to_file(out);
        moment_weights.to_file(out);
        moment_bias.to_file(out);
    }

    inline void from_file(std::ifstream &in) noexcept
    {
        weights.from_file(in);
        bias.from_file(in);
        moment_weights.from_file(in);
        moment_bias.from_file(in);
    }

    inline void to_bin_file(std::ofstream &out) const noexcept
    {
        weights.to_bin_file(out);
        bias.to_bin_file(out);
        moment_weights.to_bin_file(out);
        moment_bias.to_bin_file(out);
    }

    inline void from_bin_file(std::ifstream &in) noexcept
    {
        weights.from_bin_file(in);
        bias.from_bin_file(in);
        moment_weights.from_bin_file(in);
        moment_bias.from_bin_file(in);
    }

    inline void apply_momentum(const double momentum) noexcept
    {
        moment_weights.mul(momentum);
        moment_bias.mul(momentum);
        weights.add(moment_weights);
        bias.add(moment_bias);
    }

    inline void update_weights_momentum(const double learning_rate) noexcept
    {
        update_weight_momentum(weights, moment_weights, weight_gradient_accumulator, learning_rate);
        update_weight_momentum(bias, moment_bias, bias_gradient_accumulator, learning_rate);

        weight_gradient_accumulator.set(0.0);
        bias_gradient_accumulator.set(0.0);
    }
};

template<unsigned long input_size, unsigned long output_size>
class SpeedySoftmaxBlock : public BaseSoftmaxBlock<input_size, output_size>
{
private:
    Matrix<input_size, output_size> moment_weights;
    Matrix<1, output_size> moment_bias;
    Matrix<input_size, output_size> ms_weights;
    Matrix<1, output_size> ms_bias;
public:
    using BaseSoftmaxBlock<input_size, output_size>::layer_states;
    using BaseSoftmaxBlock<input_size, output_size>::weights;
    using BaseSoftmaxBlock<input_size, output_size>::bias;
    using BaseSoftmaxBlock<input_size, output_size>::weight_gradient_accumulator;
    using BaseSoftmaxBlock<input_size, output_size>::bias_gradient_accumulator;
    SpeedySoftmaxBlock(size_t time_steps=0) noexcept:BaseSoftmaxBlock<input_size, output_size>(time_steps),moment_weights(0.0),moment_bias(0.0),ms_weights(1.0), ms_bias(1.0)
    {
    }

    inline void to_file(std::ofstream &out) const noexcept
    {
        weights.to_file(out);
        bias.to_file(out);
        moment_weights.to_file(out);
        moment_bias.to_file(out);
        ms_weights.to_file(out);
        ms_bias.to_file(out);
    }

    inline void from_file(std::ifstream &in) noexcept
    {
        weights.from_file(in);
        bias.from_file(in);
        moment_weights.from_file(in);
        moment_bias.from_file(in);
        ms_weights.from_file(in);
        ms_bias.from_file(in);
    }

    inline void to_bin_file(std::ofstream &out) const noexcept
    {
        weights.to_bin_file(out);
        bias.to_bin_file(out);
        moment_weights.to_bin_file(out);
        moment_bias.to_bin_file(out);
        ms_weights.to_bin_file(out);
        ms_bias.to_bin_file(out);
    }

    inline void from_bin_file(std::ifstream &in) noexcept
    {
        weights.from_bin_file(in);
        bias.from_bin_file(in);
        moment_weights.from_bin_file(in);
        moment_bias.from_bin_file(in);
        ms_weights.from_bin_file(in);
        ms_bias.from_bin_file(in);
    }

    inline void apply_momentum(const double momentum) noexcept
    {
        moment_weights.mul(momentum);
        moment_bias.mul(momentum);
        weights.add(moment_weights);
        bias.add(moment_bias);
    }

    inline void update_weights_momentum_ms(const double learning_rate, const double decay) noexcept
    {
        ms_weights.mul(decay);
        ms_bias.mul(decay);
        ms_weights.add_factor_mul_a_squared(1-decay, weight_gradient_accumulator);
        ms_bias.add_factor_mul_a_squared(1-decay, bias_gradient_accumulator);
        update_weight_momentum(weights, moment_weights, ms_weights, weight_gradient_accumulator, learning_rate);
        update_weight_momentum(bias, moment_bias, ms_bias, bias_gradient_accumulator, learning_rate);

        weight_gradient_accumulator.set(0.0);
        bias_gradient_accumulator.set(0.0);
    }
};

template<unsigned long input_size, unsigned long output_size>
class RMSPropSoftmaxBlock : public BaseSoftmaxBlock<input_size, output_size>
{
private:
    Matrix<input_size, output_size> ms_weights;
    Matrix<1, output_size> ms_bias;
public:
    using BaseSoftmaxBlock<input_size, output_size>::layer_states;
    using BaseSoftmaxBlock<input_size, output_size>::weights;
    using BaseSoftmaxBlock<input_size, output_size>::bias;
    using BaseSoftmaxBlock<input_size, output_size>::weight_gradient_accumulator;
    using BaseSoftmaxBlock<input_size, output_size>::bias_gradient_accumulator;
    RMSPropSoftmaxBlock(size_t time_steps=0) noexcept:BaseSoftmaxBlock<input_size, output_size>(time_steps),ms_weights(1.0), ms_bias(1.0)
    {
    }

    inline void to_file(std::ofstream &out) const noexcept
    {
        weights.to_file(out);
        bias.to_file(out);
        ms_weights.to_file(out);
        ms_bias.to_file(out);
    }

    inline void from_file(std::ifstream &in) noexcept
    {
        weights.from_file(in);
        bias.from_file(in);
        ms_weights.from_file(in);
        ms_bias.from_file(in);
    }

    inline void to_bin_file(std::ofstream &out) const noexcept
    {
        weights.to_bin_file(out);
        bias.to_bin_file(out);
        ms_weights.to_bin_file(out);
        ms_bias.to_bin_file(out);
    }

    inline void from_bin_file(std::ifstream &in) noexcept
    {
        weights.from_bin_file(in);
        bias.from_bin_file(in);
        ms_weights.from_bin_file(in);
        ms_bias.from_bin_file(in);
    }

    inline void update_weights_ms(const double learning_rate, const double decay) noexcept
    {
        ms_weights.mul(decay);
        ms_bias.mul(decay);
        ms_weights.add_factor_mul_a_squared(1-decay, weight_gradient_accumulator);
        ms_bias.add_factor_mul_a_squared(1-decay, bias_gradient_accumulator);
        update_weight_with_ms(weights, ms_weights, weight_gradient_accumulator, learning_rate);
        update_weight_with_ms(bias, ms_bias, bias_gradient_accumulator, learning_rate);

        weight_gradient_accumulator.set(0.0);
        bias_gradient_accumulator.set(0.0);
    }
};

template<unsigned long input_size, unsigned long output_size>
class AdamSoftmaxBlock : public BaseSoftmaxBlock<input_size, output_size>
{
private:
    Matrix<input_size, output_size> ms_weights;
    Matrix<1, output_size> ms_bias;
    Matrix<input_size, output_size> mns_weights;
    Matrix<1, output_size> mns_bias;
public:
    using BaseSoftmaxBlock<input_size, output_size>::layer_states;
    using BaseSoftmaxBlock<input_size, output_size>::weights;
    using BaseSoftmaxBlock<input_size, output_size>::bias;
    using BaseSoftmaxBlock<input_size, output_size>::weight_gradient_accumulator;
    using BaseSoftmaxBlock<input_size, output_size>::bias_gradient_accumulator;
    AdamSoftmaxBlock(size_t time_steps=0) noexcept:BaseSoftmaxBlock<input_size, output_size>(time_steps),ms_weights(1.0), ms_bias(1.0),mns_weights(1.0), mns_bias(1.0)
    {
    }

    inline void to_file(std::ofstream &out) const noexcept
    {
        weights.to_file(out);
        bias.to_file(out);
        ms_weights.to_file(out);
        ms_bias.to_file(out);
        mns_weights.to_file(out);
        mns_bias.to_file(out);
    }

    inline void from_file(std::ifstream &in) noexcept
    {
        weights.from_file(in);
        bias.from_file(in);
        ms_weights.from_file(in);
        ms_bias.from_file(in);
        mns_weights.from_file(in);
        mns_bias.from_file(in);
    }

    inline void to_bin_file(std::ofstream &out) const noexcept
    {
        weights.to_bin_file(out);
        bias.to_bin_file(out);
        ms_weights.to_bin_file(out);
        ms_bias.to_bin_file(out);
        mns_weights.to_bin_file(out);
        mns_bias.to_bin_file(out);
    }

    inline void from_bin_file(std::ifstream &in) noexcept
    {
        weights.from_bin_file(in);
        bias.from_bin_file(in);
        ms_weights.from_bin_file(in);
        ms_bias.from_bin_file(in);
        mns_weights.from_bin_file(in);
        mns_bias.from_bin_file(in);
    }

    inline void update_weights_adam(const double learning_rate, const double decay1, const double decay2) noexcept
    {
        ms_weights.mul(decay1);
        ms_bias.mul(decay1);
        mns_weights.mul(decay2);
        mns_bias.mul(decay2);
        ms_weights.add_factor_mul_a_squared(1-decay1, weight_gradient_accumulator);
        ms_bias.add_factor_mul_a_squared(1-decay1, bias_gradient_accumulator);
        mns_weights.add_factor_mul_a(1-decay2, weight_gradient_accumulator);
        mns_bias.add_factor_mul_a(1-decay2, bias_gradient_accumulator);

        update_weight_with_adam(weights, ms_weights, mns_weights, learning_rate, decay1, decay2);
        update_weight_with_adam(bias, ms_bias, mns_bias, learning_rate, decay1, decay2);

        weight_gradient_accumulator.set(0.0);
        bias_gradient_accumulator.set(0.0);
    }
};