#include "perceptron_timeseries_class.hpp"
#include "LSTM_class.hpp"
#include "softmax_timeseries_class.hpp"
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <memory>
#include <ctime>
#include "mystuff.hpp"
using namespace std;

template<unsigned long input_size, unsigned long first_mem_cell_size, unsigned long second_mem_cell_size, unsigned long third_mem_cell_size, unsigned long output_mem_size>
class MyClass
{
private:
    static constexpr unsigned long reduced_input_size=input_size/4;
    using Block01=NAGTahnPerceptronBlock<input_size,reduced_input_size>;
    using Block02=NAGLSTMBlock<reduced_input_size, first_mem_cell_size>;
    using Block03=NAGLSTMBlock<first_mem_cell_size, second_mem_cell_size>;
    using Block04=NAGLSTMBlock<second_mem_cell_size, third_mem_cell_size>;
    using Block05=NAGSoftmaxBlock<third_mem_cell_size, output_mem_size>;

    unique_ptr<Block01> perceptronblock;
    unique_ptr<Block02> lstmblock1;
    unique_ptr<Block03> lstmblock2;
    unique_ptr<Block04> lstmblock3;
    unique_ptr<Block05> softmaxblock;

    OneHot<input_size> X;
    OneHot<output_mem_size> Y;
    const string &index_to_char;
    const unordered_map<char, size_t> &char_to_index;
public:
    MyClass(const string &index_to_char, const unordered_map<char, size_t> &char_to_index):perceptronblock(new Block01),lstmblock1(new Block02),lstmblock2(new Block03),lstmblock3(new Block04),softmaxblock(new Block05)
    ,index_to_char(index_to_char), char_to_index(char_to_index)
    {}

    inline void reserve_time_steps(size_t ts)
    {
        perceptronblock->reserve_time_steps(ts);
        lstmblock1->reserve_time_steps(ts);
        lstmblock2->reserve_time_steps(ts);
        lstmblock3->reserve_time_steps(ts);
        softmaxblock->reserve_time_steps(ts);
    }

    template<bool verbose=true>
    inline void pre_train()
    {
        static constexpr double learning_rate=0.1;
        static constexpr double momentum=0.9;
        static constexpr size_t batch_size=1;
        std::random_device rd;
        std::mt19937 gen(rd());
        if(verbose)print("First run... starting pre-training.");
        using BlockPre=NAGSoftmaxBlock<reduced_input_size,input_size>;
        unique_ptr<BlockPre> softmaxblock2(new BlockPre);
        softmaxblock2->set_time_steps(1);
        perceptronblock->set_time_steps(1);
        std::uniform_int_distribution<size_t> dst(0,input_size-1);

        double error=1.0;
        for(size_t iteration=0;error>0.00000001;iteration++)
        {
            perceptronblock->apply_momentum(momentum);
            softmaxblock2->apply_momentum(momentum);
            for(size_t batch=0;batch<batch_size;batch++)
            {
                X.set(dst(gen));

                perceptronblock->calc(X.get(), 0);
                softmaxblock2->calc(perceptronblock->get_output(0), 0);

                softmaxblock2->set_first_delta_and_propagate_with_cross_enthropy(X.get(), perceptronblock->get_delta_output(0), 0);
                perceptronblock->propagate_delta(0);

                perceptronblock->accumulate_gradients(X.get(),0);
                softmaxblock2->accumulate_gradients(perceptronblock->get_output(0),0);
            }
            perceptronblock->update_weights_momentum(learning_rate);
            softmaxblock2->update_weights_momentum(learning_rate);

            if(iteration%10000==0)
            {
                error=0.0;
                for(size_t new_input_index=0;new_input_index<input_size;new_input_index++)
                {
                    X.set(dst(gen));

                    perceptronblock->calc(X.get(), 0);
                    softmaxblock2->calc(perceptronblock->get_output(0), 0);

                    for(size_t i=0;i<input_size;i++)
                    {
                        double aux=softmaxblock2->get_output(0)[0][i]-X.get()[0][i];
                        aux*=aux;
                        error+=aux;
                    }
                }
                if(verbose)print(error);
            }
        }
    }

    inline void apply_momentum(double momentum)
    {
        perceptronblock->apply_momentum(momentum);
        lstmblock1->apply_momentum(momentum);
        lstmblock2->apply_momentum(momentum);
        lstmblock3->apply_momentum(momentum);
        softmaxblock->apply_momentum(momentum);
    }

    inline double training_iteration(const string &str, const size_t out_index)
    {
        perceptronblock->set_time_steps(str.size());
        lstmblock1->set_time_steps(str.size());
        lstmblock2->set_time_steps(str.size());
        lstmblock3->set_time_steps(str.size());
        softmaxblock->set_time_steps(str.size());
        for(size_t i=0;i<str.size();i++)
        {
            X.set(char_to_index.at(str[i]));

            perceptronblock->calc(X.get(), i);
            lstmblock1->calc(perceptronblock->get_output(i), i);
            lstmblock2->calc(lstmblock1->get_output(i), i);
            lstmblock3->calc(lstmblock2->get_output(i), i);
            softmaxblock->calc(lstmblock3->get_output(i), i);
        }

        Y.set(out_index);
        softmaxblock->set_first_delta_and_propagate_with_cross_enthropy(Y.get(), lstmblock3->get_delta_output(str.size()-1), str.size()-1);
        double error=softmaxblock->get_delta_output(str.size()-1).sum_of_squares();
        lstmblock3->propagate_delta(lstmblock2->get_delta_output(str.size()-1), str.size()-1, str.size());
        lstmblock2->propagate_delta(lstmblock1->get_delta_output(str.size()-1), str.size()-1, str.size());
        lstmblock1->propagate_delta(perceptronblock->get_delta_output(str.size()-1), str.size()-1, str.size());
        perceptronblock->propagate_delta(str.size()-1);
        for(size_t i=str.size()-2;;)
        {
            //Set up output
            lstmblock3->get_delta_output(i).set(0.0);
            lstmblock3->propagate_delta(lstmblock2->get_delta_output(i), i, str.size());
            lstmblock2->propagate_delta(lstmblock1->get_delta_output(i), i, str.size());
            lstmblock1->propagate_delta(perceptronblock->get_delta_output(i), i, str.size());
            perceptronblock->propagate_delta(i);
            if(i--==0)break;
        }

        for(size_t i=0;i<str.size();i++)
        {
            //Set up input
            X.set(char_to_index.at(str[i]));

            perceptronblock->accumulate_gradients(X.get(), i);
            lstmblock1->accumulate_gradients(perceptronblock->get_output(i), i);
            lstmblock2->accumulate_gradients(lstmblock1->get_output(i), i);
            lstmblock3->accumulate_gradients(lstmblock2->get_output(i), i);
            softmaxblock->accumulate_gradients(lstmblock3->get_output(i), i);
        }

        return error;
    }

    inline void update(const double learning_rate)
    {
        perceptronblock->update_weights_momentum(learning_rate);
        lstmblock1->update_weights_momentum(learning_rate);
        lstmblock2->update_weights_momentum(learning_rate);
        lstmblock3->update_weights_momentum(learning_rate);
        softmaxblock->update_weights_momentum(learning_rate);
    }

    inline void to_file(const char *filename)
    {
        ofstream out(filename,std::ios_base::trunc|std::ios::binary);
        assert(out.good());
        perceptronblock->to_bin_file(out);
        lstmblock1->to_bin_file(out);
        lstmblock2->to_bin_file(out);
        lstmblock3->to_bin_file(out);
        softmaxblock->to_bin_file(out);
    }

    inline void from_file(const char *filename)
    {
        ifstream in(filename,std::ios::binary);
        assert(in.good());
        perceptronblock->from_bin_file(in);
        lstmblock1->from_bin_file(in);
        lstmblock2->from_bin_file(in);
        lstmblock3->from_bin_file(in);
        softmaxblock->from_bin_file(in);
    }

    inline void only_wb_to_bin_file(const char *filename)
    {
        ofstream out(filename,std::ios_base::trunc|std::ios::binary);
        assert(out.good());
        perceptronblock->only_wb_to_bin_file(out);
        lstmblock1->only_wb_to_bin_file(out);
        lstmblock2->only_wb_to_bin_file(out);
        lstmblock3->only_wb_to_bin_file(out);
        softmaxblock->only_wb_to_bin_file(out);
    }
};

//maybe add a zero char at the end of each song, and throw in cross entropy error twice (remember to update str.size() everywhere)
//throw in cross entropy error all the time?
//change learning rate schedule
//check error with testing set
int main()
{
    static constexpr size_t num_songs=10;
    static const array<string, num_songs> filenames={"blink-182.txt", "queen.txt", "the_beatles.txt", "eminem.txt", "nightwish.txt",
    "frank_sinatra.txt","britney_spears.txt", "shakira.txt", "epica.txt", "metallica.txt"};
    static const char *testingfilepath="../songs/testing/%s";
    static const char *trainingfilepath="../songs/training/%s";
    static const char *savecounter_filename="data/savecounter.svc";
    static const char *savestate_filename="data/savestate%08lu.sst";
    static const char *wb_filename="data/wb.wab";
    static constexpr time_t secons_between_saves=1*60*60;

    static constexpr size_t allowed_char_amount=54;
    static const string index_to_char="abcdefghijklmnopqrstuvwxyz0123456789 \n\"`'$()/?[].:,;!-";
    unordered_map<char, size_t> char_to_index;for(size_t i=0;i<index_to_char.size();i++) char_to_index[index_to_char[i]]=i;
    assert(index_to_char.size()==allowed_char_amount and char_to_index.size()==allowed_char_amount);

    std::random_device rd;
    std::mt19937 gen(rd());
    uniform_int_distribution<size_t> song_dst(0,num_songs-1);
    MyClass<allowed_char_amount,200,100,50,num_songs> myclass(index_to_char,char_to_index);
    myclass.reserve_time_steps(8000);

    array<vector<string>, num_songs> training_songs;
    array<vector<string>, num_songs> testing_songs;
    array<uniform_int_distribution<size_t>, 10> training_dst;
    array<uniform_int_distribution<size_t>, 10> testing_dst;
    for(size_t i=0;i<num_songs;i++)
    {
        char cbuffer[256];
        string aux_string;
        sprintf(cbuffer, trainingfilepath, filenames[i].c_str());
        read_file_to_string(cbuffer, aux_string);
        training_songs[i]=split_string(aux_string, "################################");
        training_dst[i]=uniform_int_distribution<size_t>(0,training_songs[i].size()-1);
        sprintf(cbuffer, testingfilepath, filenames[i].c_str());
        read_file_to_string(cbuffer, aux_string);
        testing_songs[i]=split_string(aux_string, "################################");
        testing_dst[i]=uniform_int_distribution<size_t>(0,testing_songs[i].size()-1);
    }

    double error=0.0;
    size_t save_counter=0;
    size_t iteration=0;
    double learning_rate=0.01;
    double momentum=0.5;
    static constexpr size_t batch_size=1;
    {
        ifstream in(savecounter_filename, std::ios::binary);
        if(in.good())
        {
            in >> save_counter >> iteration >> error >> learning_rate >> momentum;
            assert(not in.fail());
            assert(save_counter!=0);
            {
                char cbuffer[256];
                sprintf(cbuffer, savestate_filename, save_counter);
                myclass.from_file(cbuffer);
            }
        }
        else
        {
            myclass.pre_train();
        }
    }

    time_t last_time=time(nullptr);
    for(;;iteration++)
    {
        if(iteration%1000==0)
        {
            learning_rate=0.01*pow(0.9549925860214360, (iteration/1000));// gets divided by 10 every 50k steps
            if(iteration<=50000) momentum=0.5+0.008*(iteration/1000);
            else if(iteration<=100000)momentum=0.9+0.0018*((iteration-50000)/1000);
            // else if(iteration<=150000)momentum=0.99+0.00018*((iteration-100000)/1000);
            // else if(iteration<=200000)momentum=0.999+0.000018*((iteration-150000)/1000);
            else momentum=.99;
        }
        myclass.apply_momentum(momentum);
        error*=0.999;
        for(size_t batch=0;batch<batch_size;batch++)
        {
            size_t out_index=song_dst(gen);
            error+=(myclass.training_iteration(training_songs[out_index][training_dst[out_index](gen)], out_index)/batch_size)*0.001;
        }
        error/=batch_size;
        myclass.update(learning_rate);

        if(time(nullptr)-last_time>secons_between_saves)
        {
            last_time=time(nullptr);
            print("Saving current state...");
            save_counter++;
            myclass.only_wb_to_bin_file(wb_filename);

            char cbuffer[256];
            sprintf(cbuffer, savestate_filename, save_counter);
            myclass.to_file(cbuffer);
            {
                ofstream out(savecounter_filename,std::ios_base::trunc);
                assert(out.good());
                out << save_counter << "\t" << iteration << "\t" << error << "\t" << learning_rate << "\t" << momentum <<endl;
            }
            print("State with number", save_counter, "saved");
        }
        print("Iteration:", iteration, "Error:", error);
    }

    return 0;
}