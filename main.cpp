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

const int CREATURE_COUNT = 36;
const int EPOCHES = 25;
const double SURVIVAL_RATE = 0.5;

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

int ValueToIndex(std::vector<double> list, double value)
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

//creature compuation
void CreatureComputation(std::vector<Brainz::LSTM*> *creatures,nlohmann::json TrainingData,std::vector<double>* errors,int nc)
{
  //loop through inputs and outputs
      for(int i = 0; i != TrainingData["Input"].size();i++)
      {
        //degrammarlize and form ints of inputs
        std::vector<int> inps;
        inps = TextToInts(TrainingData["Input"][i]);

        //degrammarlize and form ints of outputs
        auto outs = TextToInts(TrainingData["Output"][i]);

        //loop inputs through network
        for(int num = 0; num != inps.size();num++)
        {
          creatures->at(nc)->Run(inps[num]);
        }


        //loop through outputs and calculate errors
        for(int num = 0; num != outs.size()-1;num++)
        {

          //run network
          double r = creatures->at(nc)->Run((double)outs[num]);

          r = r*1000.0;

          //run network on next input for error check
          double nr = ((double)outs[num+1]);

          //if first time error is being added, append to error, else just add to existing
          while(errors->size() < nc)
          {
            std::cout<<"";
          }
          if(errors->size() < nc)
          {
              std::cout<<"Hmm...Recovering code... \n";
          }
          if(errors->size() < nc)
          {
              std::cout<<"Last ditch effort to recover... \n";
          }
          if(errors->size() < nc)
          {
              std::this_thread::sleep_for(std::chrono::seconds(1));
          }
          if(errors->size() < nc)
          {
              std::this_thread::sleep_for(std::chrono::seconds(10));
          }
          if(errors->size() == nc)
          {
            errors->push_back(fabs(r - nr));
          }
          else
          {
            errors->at(nc) += fabs(r - nr);
          }
        }
      }
}

void CreateMoreCreatures(int BaseNum,std::vector<Brainz::LSTM*> *creatures,nlohmann::json TrainingData,std::vector<double>* errors)
{
  std::vector<std::thread*> threads;
  //create creatures with survied creatures
  for(int nc = BaseNum-1;nc != CREATURE_COUNT -1; nc++)
  {
      //create creature
      Brainz::LSTM* creature = new Brainz::LSTM();

      //get json data of a survived creture
      auto j = creatures->at(nc % BaseNum)->Save();

      int seed = creatures->at(nc % BaseNum)->GetSeed();

      //load data
      creature->Load(j);

      const auto p1 = std::chrono::system_clock::now();

      int unix = (int)std::chrono::duration_cast<std::chrono::nanoseconds>(p1.time_since_epoch()).count();

      //set random seed
      creature->SetSeed(seed,nc+unix);

      //mutate creture
      creature->Mutate();

      //add creture to envirement
      creatures->push_back(creature);

      //make thread for creature computation
      std::thread* t = new std::thread(CreatureComputation,creatures,TrainingData,errors,nc);
      threads.push_back(t);

  }


    //create creature
    Brainz::LSTM* creature = new Brainz::LSTM();

    //mutate creture
    creature->Generate();

    //add creture to envirement
    creatures->push_back(creature);

    //make thread for creature computation
    std::thread* t = new std::thread(CreatureComputation,creatures,TrainingData,errors,CREATURE_COUNT -1);
    threads.push_back(t);

  //reconnect threads
  for(int i = 0; i != threads.size();i++)
  {
    threads[i]->join();
  }
}

//train bot through cops vs robbers
void CopsVsRobbers(Brainz::LSTM Cop,Brainz::LSTM Robber)
{
  //loop through epoches
  for(int e = 0; e != EPOCHES; e++)
  {
    
  }
}


int main() {
  //create bots
  Brainz::LSTM cop;
  Brainz::LSTM robber;

  //load bot brain matrix
  std::ifstream file;

  //make/load cop
  file.open("bot/Cop-Brain.json");
  try
  {
    auto j = nlohmann::json::parse(file);
    cop.Load(j);
  }catch(...){
    cop.Generate();
  }

  //make/load robber
  file.open("bot/Robber-Brain.json");
  try
  {
    auto j = nlohmann::json::parse(file);
    cop.Load(j);
  }catch(...){
    cop.Generate();
  }
  
  file.close();
  
  //load Robber and Cop training data
  nlohmann::json RobberTrainingData;
  nlohmann::json CopTrainingData;
  std::ifstream data;
  std::ifstream data2;
  data.open("Training/Robber-TrainingData.json");
  data.open("Training/Cop-TrainingData.json");
  try
  {
    RobberTrainingData = nlohmann::json::parse(data);
    CopTrainingData = nlohmann::json::parse(data);
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

  if (ch[0] == 'Y' || ch[0] == 'y')
  {
    RobberTrainingData = Reload(0,"robber");
    CopTrainingData = Reload(0,"cop");
  }
  
  auto g = TextToInts("What are you doing!");

  auto inp = TextToInts("It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him.");

  //auto c = NaturalSelection(cop,CopTrainingData);

  auto j = c->Save();

  bot.Load(j);

  double out;

  for(int i = 0; i != inp.size();i++)
  {
    out = bot.Run(inp[i]);
    out *= 1000.0;
  }

  std::vector<int> sentence;
  //have bot print out 10 words
  for(int i = 0; i != 100; i++)
  {
    out = bot.Run(out);
    out *= 1000.0;
    int outs =  (int)out % 256;

    sentence.push_back(outs);
  }

  //convert ints from lstm to text
  std::string s = IntsToText(sentence);
  std::cout<<s<<std::endl;


  //save network
  std::ofstream ofile;
  ofile.open("Training/TrainingData.json");
  ofile << TrainingData;
  ofile.close();

  //save network
  ofile.open("bot/brain.json");
  auto save = bot.Save();
  ofile << save;
  ofile.close();
  
}