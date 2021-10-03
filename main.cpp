#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif


#include <iostream>
#include "brainz/brainz.hpp"
#include <fstream>
#include <string>
#include <vector>
#include "brainz/nlohmann/json.hpp"
#include <map>
#include <cmath>
#include <thread>
#include <sys/stat.h>
#define ROUNDNUM(x) ((int)(x + 0.5f));



const int CREATURE_COUNT = 100;
const int EPOCHES = 10000;
const double SURVIVAL_RATE = 0.5;
const int TOLERANCE = 0;

/*
  Training Plan:

  Load all needed data

  degrammarlize input text (make each character into ascii version)

  Use cop vs robber method
    -Make "robber" ai
    -Make "cop" ai
    -Train cop to identify stories made by bot
    -Train robber to make stories that trick the cop
    -Repeat

  save brain matrix
*/
//check if file exist(Credit: PherricOxide)
inline bool exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}


//loop through training files and collect training data recursuvly
nlohmann::json Reload(int start,std::string filename)
{
  //file stream
  std::ifstream ifile;
  nlohmann::json j;


  //open file
  ifile.open("Training/" + std::to_string(start) + filename+".txt");

  if(ifile.is_open()){

    //start next instance of coleccting data
    j = Reload(start + 1,filename);

    //loop vars
    std::string line;
    std::string sentence;
    std::vector<std::string> sentences;

    while(std::getline(ifile,line))
    {
      //loop through lines and break up into sentences
      for(int i = 0; i != line.size();i++)
      {
        //verify it's not the end of a sentence
        if(line[i] != '.' && line[i] != '!' && line[i] != '?')
        {
          //add character to sentence
          sentence = sentence + line[i];
        } else
        {
          //add chacter to sentence
          sentence = sentence + line[i];
          //add sentence to array
          sentences.push_back(sentence);
          //reset sentence
          sentence = "";
        }
      }
      //close file
      ifile.close();

      //Make training data
      for(int i = 0; i != sentences.size() && sentences.size() >= 4;i++)
      {
        //if on good index, start making input and output data
        if((i+1) >= 4)
        {
          int index = j["Input"].size();
          //make input data
          j["Input"][index] = sentences[i-3]+sentences[i-2];

          //make output data
          j["Output"][index] = sentences[i-1]+sentences[i];
        }
      }
    }
    return j;
    
  }
  else
  {
    //make defualt training data
    j["Input"][0] = "Hi! How are you?";

    j["Output"][0] = "My Name is StoryBotAI and I'm happy to be of service!";
    return j;
  }
}

//convert string to ints
std::vector<int> TextToInts(std::string text)
{
  std::vector<int> output;

  //loop through words and symbols
  for(int i = 0; i != text.size();i++)
  {
    //add int version of word to output
    output.push_back(text[i]);
  }

  return output;
}

//convert ints to string
std::string IntsToText(std::vector<int> text)
{
  int max = 256;
  std::string output;

  for(int i = 0; i != text.size();i++)
  {
    int w  = (int)(abs(text[i]) % max);
    if(abs(w) >= max)
    {
     output += "[Error]";
    }
    else{
      output += (char)text[i];
    }
  }

  return output;
}

template<typename T>
int ValueToIndex(std::vector<T> list, T value)
{
  //loop through list
  for(int i = 0; i != list.size();i++)
  {
    if (list[i] == value) return i;
  }

  return -1;
}

//merge sort algorithm
std::vector<double> MergeSort(std::vector<double> x)
{
  //if array size is 1 just return self
  if (x.size() <= 1)
  {
    return x;
  }

  //cut in half and get start and stop
  int x1 = (int)(x.size() / 2);
  int x2 = x.size() - x1;

  std::vector<double> a1;

  //make temp array1
  for (int i = 0; i != x1;i++)
  {
    a1.push_back(x[i]);
  }

  std::vector<double> a2;

  //make temp array2
  for (int i = x1; i != x.size();i++)
  {
    a2.push_back(x[i]);
  }

  //mergesorts
  std::vector<double> m1 = MergeSort(a1);
  std::vector<double> m2 = MergeSort(a2);

  //array counters
  int m1x = 0;
  int m2x = 0;


  //merged array
  std::vector<double> merged;

  //loop through arrays and compares
  while (m1x != m1.size() and m2x != m2.size())
  {
    if (m1[m1x] >= m2[m2x])
    {
      merged.push_back(m2[m2x]);
      m2x++;
    }else{
      merged.push_back(m1[m1x]);
      m1x++;
    }
  }

//loop through any remaining vars
while (m1x != m1.size() or m2x != m2.size()){
  //add last var to merged
  if (m1x != m1.size())
  {
    merged.push_back(m1[m1x]);
    m1x++;
  }else{
    merged.push_back(m2[m2x]);
    m2x++;
  }
}
  return merged;
};

//error computatauion thread for training cop
void ComputeErrors(std::vector<double>* Errors,nlohmann::json TrainingData,std::vector<Brainz::LSTM*> *bots, int index)
{
  //loop through training data
  for(int i = 0; i != TrainingData["Input"].size();i++)
  {
    //convert text to ints
    auto inps = TextToInts(TrainingData["Input"][i]);

    double out = 0;

    //loop though inputs for training
    for(int ins = 0; ins != inps.size();ins++)
    {
      out = bots->at(index)->Run((double)inps[ins]);
    }

    //get expected output and compare
    double recived = out;
    int expected = TrainingData["Output"][i];
    double error = fabs(out - expected);

    //add error to error array
    Errors->at(index) += error;
  }
}

void RunRobberComputation(std::vector<double>* Errors,nlohmann::json RobberTrainingData,nlohmann::json* CopTrainingData,std::vector<Brainz::LSTM*> *bots,Brainz::LSTM cop, int index)
{
  //loop through training data
  for(int i = 0; i != RobberTrainingData["Inputs"].size();i++)
  {
    auto inps = TextToInts(RobberTrainingData["Inputs"][i]); 
    auto outs = TextToInts(RobberTrainingData["Output"][i]); 
    std::vector<int> sentence;

    double out;
    int answer;

    //loop though inputs for training
    for(int ins = 0; ins != inps.size();ins++)
    {
      out = bots->at(index)->Run((double)inps[ins]);
    }

    //make sentence
    for(int ous = 0; ous != outs.size();ous++)
    {
      out = bots->at(index)->Run(out);
      sentence.push_back((int)(out*1000));
      answer = ROUNDNUM(cop.Run(out));
    }
    if ((answer != 1))
    {
      CopTrainingData->at("Inputs").push_back(IntsToText(sentence));
      CopTrainingData->at("Output").push_back(1);
    }
    else
    {
      Errors->at(index) += 1;
    }
  }
}

//train bot through cops vs robbers
void CopsVsRobbers(Brainz::LSTM* Cop,Brainz::LSTM* Robber,nlohmann::json* CopTrainingData,nlohmann::json* RobberTrainingData)
{
  std::vector<Brainz::LSTM*> cops;
  std::vector<Brainz::LSTM*> robbers;
  std::vector<double>* Errors = new std::vector<double> ;
  std::vector<std::thread*> threads;

  cops.push_back(Cop);
  robbers.push_back(Robber);
  Errors->push_back(0);

  //loop through epoches
  for(int e = 0; e != EPOCHES; e++)
  {
    bool CopWon = false;
    bool RobberWon = false;
    std::vector<Brainz::LSTM*> temp;
    std::cout<<"Epoch: "<<e<<std::endl;

    std::cout<<"Training Cop..."<<std::endl;

    //Loop through and train the cop until it wins
    while(!CopWon)
    { 
      //make cops
      for(int c = cops.size(); c != CREATURE_COUNT;c++)
      {
        //make cop
        Brainz::LSTM* cop =  new Brainz::LSTM();

        //use survived creatures as base
        auto j = cops[c % cops.size()]->Save();
        cop->Load(j);

        //mutate
        cop->Mutate();

        //add to cops
        cops.push_back(cop);

        //make error partition
        Errors->push_back(0);
      }

      //run cops on trainging data and collect error
      for(int c = 0; c != cops.size();c++)
      {
        //std::vector<double>* Errors,nlohmann::json TrainingData,std::vector<Brainz::LSTM*> *bots, int index
        std::thread* t = new std::thread(ComputeErrors,Errors,*CopTrainingData,&cops,c);

        threads.push_back(t);
      }

      //rejoin threads
      for(int t = 0; t != threads.size();t++)
      {
        threads[t]->join();
      }

    
      //sort errors
      auto sorted = MergeSort(*Errors);

      std::vector<double> dups;
    
      //kill off creatures that didn't survive or are duplicates
      for(int s = 0; s != (int)(sorted.size()*SURVIVAL_RATE);s++)
      {
        //check if it is a duplicate
        if(ValueToIndex(dups,sorted[s]) != -1)
        {
          //if it is not a duplicate, add to duplication array and add creature to next generation
          int ind = ValueToIndex(*Errors,sorted[s]);
          dups.push_back(sorted[s]);
          temp.push_back(cops[ind]);
        }
      
      }
      //set new cop array
      cops = temp;

      //reset
      Errors = new std::vector<double>;
      threads = {};
      int ro = ROUNDNUM(sorted[0])

      if( ro <= TOLERANCE)
      {
        //set cop data and exit loop
        CopWon = true;
        auto j = cops[0]->Save();
        Cop->Load(j);
      }
    }
    std::cout<<"Training Robber..."<<std::endl;
    robbers.push_back(Robber);
    //train robber
    while(!RobberWon)
    {
      //create more robbers
      for(int r = robbers.size(); r != CREATURE_COUNT; r++)
      {
        //make robber
        Brainz::LSTM* robber =  new Brainz::LSTM();

        //use survived creatures as base
        auto j = cops[r % robbers.size()]->Save();
        robber->Load(j);

        //mutate
        robber->Mutate();

        //add to cops
        cops.push_back(robber);

        //make error partition
        Errors->push_back(0);
      }
      
      //run through compuation an get errors
      for(int c = 0; c != robbers.size();c++)
      {
        //std::vector<double>* Errors,nlohmann::json RobberTrainingData,nlohmann::json* CopTrainingData,std::vector<Brainz::LSTM*> *bots,Brainz::LSTM cop, int index
        std::thread* t = new std::thread(RunRobberComputation,Errors,*RobberTrainingData,CopTrainingData,&robbers,*Cop,c);
        threads.push_back(t);
      }

      //re join threads
      for(int t = 0; t != threads.size();t++)
      {
        threads[t]->join();
      }
      
      //sort data
      auto sorted = MergeSort(*Errors);

      /*
      for(int s = 0; s != sorted.size();s++)
      {
        std::cout<<sorted[s] << " ";
      }
      std::cout<<"\n";
      std::cout<<sorted.size()<<"\n";
      */

      std::vector<Brainz::LSTM*> temp;
      std::vector<double> dups;
    
      //kill off creatures that didn't survive or are duplicates
      for(int s = 0; s != (int)(sorted.size()*SURVIVAL_RATE);s++)
      {
        //check if it is a duplicate
        if(ValueToIndex(dups,sorted[s]) != -1)
        {
          //if it is not a duplicate, add to duplication array and add creature to next generation
          int ind = ValueToIndex(*Errors,sorted[s]);
          dups.push_back(sorted[s]);
          temp.push_back(robbers[ind]);
        }
      
      }
      
      //set new cop array
      robbers = temp;

      //reset
      Errors = new std::vector<double>;
      threads = {};
      int ro = ROUNDNUM(sorted[0])
      

      if( ro <= TOLERANCE)
      {
        //set cop data and exit loop
        RobberWon = true;
        auto j = robbers[0]->Save();
        Robber->Load(j);
      }

    }
    

      std::vector<std::string> dupsI;
      std::vector<int> dupsO;
      //clean up training data (remove duplicates)
      for(int t = 0; t != CopTrainingData->at("Inputs").size();t++)
      {
        if(ValueToIndex(dupsI,CopTrainingData->at("Inputs")[t].get<std::string>()) == -1)
      {
          dupsI.push_back(CopTrainingData->at("Inputs")[t]);
          dupsO.push_back(CopTrainingData->at("Output")[t]);
      }
      }
      CopTrainingData->at("Inputs") = dupsI;
      CopTrainingData->at("Output") = dupsO;
  }
}


int main() {
  //create bots
  Brainz::LSTM *cop = new Brainz::LSTM();
  Brainz::LSTM *robber = new Brainz::LSTM();


  //make/load cop
  if(exists("bot/CopBrain.json"))
  {
    //load bot brain matrix
    std::ifstream file("bot/CopBrain.json");
    auto j = nlohmann::json::parse(file);
    cop->Load(j);
    file.close();
  }
  else
  {
    cop->Generate();
  }

  //make/load robber
  if(exists("bot/RobberBrain.json"))
  {
    //load bot brain matrix
    std::ifstream file("bot/RobberBrain.json");
    auto j = nlohmann::json::parse(file);
    robber->Load(j);
    file.close();
  }
  else
  {
    robber->Generate();
  }
  
  //load Robber and Cop training data
  nlohmann::json RobberTrainingData;
  nlohmann::json CopTrainingData;
  std::ifstream data;
  std::ifstream data2;
  data.open("Training/RobberTrainingData.json");
  data2.open("Training/CopTrainingData.json");
  try
  {
    RobberTrainingData = nlohmann::json::parse(data);
    CopTrainingData = nlohmann::json::parse(data2);
  }
  catch(...)
  {
    RobberTrainingData = Reload(0,"robber");
    CopTrainingData = Reload(0,"cop");
  }
  data.close();

  std::string ch;

  std::cout<<"Want to reload training data?[Y|N]--> ";
  std::cin >> ch;

  /*if (ch[0] == 'Y' || ch[0] == 'y')
  {
    RobberTrainingData = Reload(0,"robber");
    CopTrainingData = Reload(0,"cop");
  }*/

  //train robber and cop ai
  CopsVsRobbers(cop,robber,&CopTrainingData,&RobberTrainingData);
  
  std::vector<int> sentenceI;

  auto ini = TextToInts("Hi! how have you been?");

  //initiale network
  for(int i = 0;i != ini.size();i++)
  {
    robber->Run(ini[i]);
  }

  double out = (double)ini[ini.size()-1]; 

  //make a sentence
  for(int i = 0;i != 20;i++)
  {
    out = robber->Run(out);

    auto o = ROUNDNUM((out*1000));

    sentenceI.push_back(o);
  }

  std::cout<<IntsToText(sentenceI)<<"\n";

  

    std::vector<std::string> dupsI;
    std::vector<int> dupsO;
    //clean up training data (remove duplicates)
    for(int t = 0; t != CopTrainingData["Inputs"].size();t++)
    {
      if(ValueToIndex(dupsI,CopTrainingData["Inputs"][t].get<std::string>()) == -1)
      {
        dupsI.push_back(CopTrainingData["Inputs"][t]);
        dupsO.push_back(CopTrainingData["Output"][t]);
      }
    }
    CopTrainingData["Inputs"] = dupsI;
    CopTrainingData["Output"] = dupsO;


  //save Trianing datas
  std::ofstream ofile;
  ofile.open("Training/CopTrainingData.json");
  ofile << CopTrainingData;
  ofile.close();

  ofile.open("Training/RobberTrainingData.json");
  ofile << RobberTrainingData;
  ofile.close();

  //save networks
  ofile.open("bot/CopBrain.json");
  auto save = cop->Save();
  ofile << save;
  ofile.close();

  ofile.open("bot/RobberBrain.json");
  save = robber->Save();
  ofile << save;
  ofile.close();
}