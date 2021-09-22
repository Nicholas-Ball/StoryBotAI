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
#include <mutex>

std::mutex m;

const int CREATURE_COUNT = 100;
const int GENERATIONS = 250;
const double SURVIVAL_RATE = 0.5;

/*
  Training Plan:

  Load all needed data

  degrammarlize input text (space out periods, make all letters lowercase, etc)

  convert text to numbers

  run natutal selection with text as input data

  save brain matrix
*/

std::map<char,char> converter = {};

std::vector<char> symbols = {'.',',','?','!','"','(',')','[',']','{','}','/','\\','@','#','$','%','^','&','*','-','_','+','=','`','~','|','<','>'};

nlohmann::json Vocab;

//checks if cahr is symbol
bool IsSymbol(char letter)
{
  bool out = false;

  //loop throught symbols and chech if it's a symbol
  for(int i = 0;i != symbols.size() && out == false;i++)
  {
    if(symbols[i] == letter)
    {
      out = true;
    }
  }

  return out;
}

char LowerCase(char letter)
{
  if (converter.find(letter) != converter.end())
  {
    letter = converter[letter];
  }
  return letter;
}

char UpperCase(char letter)
{
  //loop through coversion map
  for (auto& it : converter) {
      if (it.second == letter) {
        letter = it.first;
      }
  }

  return letter;
}

//space out commas, periods, etc
std::vector<std::string> Degrammarlize(std::string text)
{
  std::vector<std::string> output;

  //get size of input text
  int s = text.size();

  //word var for use of adding to array
  std::string word;

  //loop through characters of text and make output array of words and symbols
  for(int i = 0; i != s; i++)
  {
    //get letter
    char letter =  text[i];

    //make letter lower case
    letter = LowerCase(text[i]);

    //check if word is a white space, if so add the word to array and reset word var
    if (letter == ' ' or letter == '\n')
    {
      if(word != "")
      {
        output.push_back(word);
      }
      word = "";
    }
    //if word is a symbol (",:,;,.,{ ,[ ,etc), add word to array and symbole to array as string
    else if (IsSymbol(letter))
    {
      output.push_back(word);
      word = "";
      output.push_back(std::string(1,letter));
    } 
    //if letter, append to word
    else
    {
      word = word + letter;
    }
    
  }

  return output;
}

//run creatures through envierments


int ConvertToNumber(std::string word)
{
  //loop through vocabulary and return index
  for(int i = 0;i != Vocab["Vocabulary"].size();i++)
  {
    if (Vocab["Vocabulary"][i] == word)
    {
      return i;
    }
  }
  
  //add word to list
  Vocab["Vocabulary"][Vocab["Vocabulary"].size()] = word;
  return Vocab["Vocabulary"].size()-1;
}


std::vector<int> TextToInts(std::vector<std::string> text)
{
  std::vector<int> output;

  //loop through words and symbols
  for(int i = 0; i != text.size();i++)
  {
    //add int version of word to output
    output.push_back(ConvertToNumber(text[i]));
  }

  return output;
}

std::vector<std::string> IntsToText(std::vector<int> text)
{
  int max = Vocab["Vocabulary"].size();
  std::vector<std::string> output;

  for(int i = 0; i != text.size();i++)
  {
    int w  = (int)(abs(text[i]) % max);
    if(abs(w) >= max)
    {
     output.push_back("[Error]");
    }
    else{
      const int sel = abs(w);
      
      output.push_back(Vocab["Vocabulary"][sel]);
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
        inps = TextToInts(Degrammarlize(TrainingData["Input"][i]));

        //degrammarlize and form ints of outputs
        auto outs = TextToInts(Degrammarlize(TrainingData["Output"][i]));

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
          r = r*100000.0;

          r += (10000000000000.0 * (r == 0));


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





int main() {

  //initialize converter
  converter.insert(std::pair<char,char>('A','a'));
  converter.insert(std::pair<char,char>('B','b'));
  converter.insert(std::pair<char,char>('C','c'));
  converter.insert(std::pair<char,char>('D','d'));
  converter.insert(std::pair<char,char>('E','e'));
  converter.insert(std::pair<char,char>('F','f'));
  converter.insert(std::pair<char,char>('G','g'));
  converter.insert(std::pair<char,char>('H','h'));
  converter.insert(std::pair<char,char>('I','i'));
  converter.insert(std::pair<char,char>('J','j'));
  converter.insert(std::pair<char,char>('K','k'));
  converter.insert(std::pair<char,char>('L','l'));
  converter.insert(std::pair<char,char>('M','m'));
  converter.insert(std::pair<char,char>('N','n'));
  converter.insert(std::pair<char,char>('O','o'));
  converter.insert(std::pair<char,char>('P','p'));
  converter.insert(std::pair<char,char>('Q','q'));
  converter.insert(std::pair<char,char>('R','r'));
  converter.insert(std::pair<char,char>('S','s'));
  converter.insert(std::pair<char,char>('T','t'));
  converter.insert(std::pair<char,char>('U','u'));
  converter.insert(std::pair<char,char>('V','v'));
  converter.insert(std::pair<char,char>('W','w'));
  converter.insert(std::pair<char,char>('X','x'));
  converter.insert(std::pair<char,char>('Y','y'));
  converter.insert(std::pair<char,char>('Z','z'));

  //create bot
  Brainz::LSTM bot;

  //load bot brain matrix
  std::ifstream file;

  file.open("brain.json");
  try
  {
    auto j = nlohmann::json::parse(file);
    bot.Load(j);
  }catch(...){
    bot.Generate();
  }
  
  file.close();

  //load bot vocabulary
  std::ifstream vocabfile;
  vocabfile.open("vocabulary.json");
  Vocab = nlohmann::json::parse(vocabfile);
  vocabfile.close();


  //load training data
  nlohmann::json TrainingData;
  std::ifstream data;
  data.open("trainingdata.json");
  TrainingData = nlohmann::json::parse(data);
  data.close();

  bool loop = true;
  //prompt to add input data and expected output
  while(loop)
  {
    std::string input;
    std::string output;

    std::cout << ("Enter input(hit return to continue)--> ");
    std::getline(std::cin,input);

    //check if user wants to end
    if(input == "")
    {
      loop = false;
      break;
    }

    std::cout << ("\nEnter expected output(hit return to continue)--> ");
  
    std::getline(std::cin,output);

    //check if user wants to end
    if(output == "")
    {
      loop = false;
      break;
    }

    //set training data information if they want to add data
    TrainingData["Input"][TrainingData["Input"].size()] = input;

    TrainingData["Output"][TrainingData["Output"].size()] = output;
  }

  //save training data to learning data json
  std::ofstream td;
  td.open("trainingdata.json");
  td << TrainingData;
  td.close();

  std::vector<Brainz::LSTM*> creatures;

  creatures.push_back(&bot);

  //train bot through natrual selection
  //loop through generations
  for(int g = 0; g != GENERATIONS;g++)
  {

    //result vector 
    std::vector<double> result;

    //number of survived cretures
    int BaseNum = creatures.size();

    std::vector<double> errors;

    std::vector<std::thread*> threads;

    //loop through survived creatures and get results and errors 
    //std::vector<Brainz::LSTM*> *creatures,nlohmann::json TrainingData,std::vector<double>* errors,int nc
    for(int sc = 0; sc != BaseNum;sc++)
    {
      //create thread
      std::thread* t = new std::thread(CreatureComputation,&creatures,TrainingData,&errors,sc);

      //add to threads
      threads.push_back(t);
    }

    //rejoin threads
    for(int i = 0; i != threads.size();i++)
    {
      threads[i]->join();
    }

    //create the rest of the creatures
    CreateMoreCreatures(creatures.size(),&creatures,TrainingData,&errors);

    //sort errors
    auto sorted = MergeSort(errors);

    /*
    for(int i = 0; i != sorted.size();i++)
    {
      std::cout<< sorted[i] <<" ";
    }
    /**/
    std::cout<<"Generation "<<g<<" score: " <<sorted[0]<<"\n";

    //used error scores to remove duplicates
    std::vector<double> dups;

    //temp blank creature array
    std::vector<Brainz::LSTM*> temp;

    //check for duplicates
    bool IsDup = false;

    int survivable = (int)((double)sorted.size() * SURVIVAL_RATE);

    //kill creatures that didn't survive or are duplicated
    for(int i = 0; i != sorted.size();i++)
    { 
      //get score
      double score = sorted[i];

      //check if duplicate
      for(int d = 0; d != dups.size();d++)
      {
        if (fabs(dups[d] - score) <= 1)
        {
          IsDup = true;
          break;
        }
      }

      //if not duplicated score, add to survuved creatures
      if(IsDup == false && dups.size() != survivable)
      {
        //get creature's index
        int creatureindex = ValueToIndex(errors,score);

        //add to survived array
        temp.push_back(creatures[creatureindex]);

        dups.push_back(score);

      }
      //reset duplication bool
      IsDup = false;

    }

    //set survived creatures
    creatures = temp;
  }

  auto j = creatures[0]->Save();

  //start bot with starter sentence
  auto g = Degrammarlize("Hi!");

  auto inp = TextToInts(g);

  bot.Load(j);

  double out;

  for(int i = 0; i != inp.size();i++)
  {
    out = bot.Run(inp[i]);
    out *= 100000.0;
  }

  std::vector<int> sentence;
  //have bot print out 10 words
  for(int i = 0; i != 10; i++)
  {
    out = bot.Run(out);
    out *= 100000.0;

    sentence.push_back((int)out);
  }


  std::vector<std::string> s = IntsToText(sentence);
  for(int i = 0; i != s.size();i++)
  {
    std::cout<<s[i]<<" ";
  }
  


  //save vocabulary
  std::ofstream ofile;
  ofile.open("vocabulary.json");
  ofile << Vocab;
  ofile.close();

  //save network
  ofile.open("brain.json");
  auto save = creatures[0]->Save();
  ofile << save;
  ofile.close();


}