#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include <string>
#include "nlohmann/json.hpp"


//Neuron network
class Neuron{
  private:
    //Neuron vars
    //----------------------
    std::string name;
    //Output var
    double Output;
    //error var
    double Error;
    //bias number var
    double Bias;
    //weights
    std::vector<double> Weights;
    //input neuron names for computation
    std::vector<int> Inputs;
    //Is it an input Neuron
    bool IsInputNeuron = false;
    //Is it an Output Neuron
    bool IsOutputNeuron = false;
    //neuron activation type
    int type;
    //generate random seed
    double seed = 0;
    //pointer to network
    std::vector<Neuron*> NetworkPointer;


  public:
    //Constructor
    Neuron();

    //Compute data for the neuron
    void Compute(std::vector<double> inputs);

    //Add Neuron index to neuron to use in computation
    void AddInputNeuron(int n);

    //Set neuron is in the first layer in should take an input
    void SetAsFirstNeuron(int num);

    //Get type of neuron
    int GetType();

    //Set type of neuron
    void SetNeuronType(int n);

    //output accessor
    double GetOutput();

    //Generate random number
    double Rand(int min,int max);

    //set network pointer
    void SetPointer(std::vector<Neuron*> net);

    //set name of neuron
    void SetName(std::string name);

    //Save neuron as json and return that data
    nlohmann::json Save();

    //Load json data to neuron
    void Load(nlohmann::json j);

    //Accecor to check if neuron is an output neuron
    bool IsOutput();

    //Set neuron as output neuron
    void SetAsOutput();

    //Randomize 
    void RandomizeWeight(int seed, int WeightNum);

    //get number of weights in neuron
    int GetNumWeights();

    void RandomizeBias(int seed);

    //destruct
    ~Neuron();

};