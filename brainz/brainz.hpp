#include <iostream>
#include <vector>
#include "Neuron.hpp"
#include "nlohmann/json.hpp"

#ifndef BRAINZ
#define BRAINZ


//make brainz class 
class Brainz{
    //Basic Deep learning class deculation
    public:
    class Basic{
      private:
        //network array
        std::vector<Neuron *> network;

        //output names array
        std::vector<std::string> names;

        //randomizer seed
        int seed;
      public:
        //Generate network and save to self
        void Generate(int Columns, std::vector<int> ColumnMatrix,int type,int numberofinputs,std::vector<std::string> OutputNames);

        //Run network
        std::map<std::string,double> Run(std::vector<double> inp);

        //Save network as Json
        nlohmann::json Save();

        //Load network with Json
        void Load(nlohmann::json j);

        //natrual selection based training
        Brainz::Basic NatrualSelection(Brainz::Basic BaseNetwork,int NumCreatures,int NumGenerations,double MutationChance,double SurvivalRate,nlohmann::json InputsWithExpectedOuputs);


        //returns size of network
        int GetSize();

        //get a neuron
        Neuron* GetNeuron(int num);

        //Deconstructor
        ~Basic();
   }; 

   class LSTM{
     private:
        //network array
        std::map<std::string,Neuron *> network;

        //randomizer seed
        int seed;

        //output value
        double output;

        //memory
        double memory;
     public:

     //constructor
      LSTM(int SeedOffset = 0);

      //Generate LSTM
      void Generate();

      //Run LSTM
      double Run(double input);

      //save lstm
      nlohmann::json Save();

      //load lstm
      void Load(nlohmann::json);

      //mutate lstm
      nlohmann::json Mutate();

      //gets seed of lastm
      int GetSeed();

      //set seed or lstm
      void SetSeed(int seed,int offset);



   };

    //Deconstructor 
    ~Brainz();
};

#endif