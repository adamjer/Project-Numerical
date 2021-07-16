/*
	Made in aKaDeMiK by Adam Jereczek
	Indeks: 149448
	*/

#include<cstdio>
#include<ctime>
#include<cstdlib>
#include<vector>


#define PLAYERS_NUMBER		2
#define WALLS_ON_DICE		6
#define MAX_GAUSS_SEIDELL	1000
#define MAX_LOTTERY_GAMES	1000000


enum playersID
{
	PLAYER_ONE,
	PLAYER_TWO
};


class GameBoard
{
public:
	std::vector<int> vector;

	GameBoard(unsigned int rozmiar)
	{
		for (unsigned int i = 0; i < rozmiar; ++i)
		{
			this->vector.push_back(0);
		}
	}

	void display()
	{
		printf("Board view!\n");
		for (unsigned int i = 0; i < this->vector.size(); ++i)
		{
			printf("|");
			printf("%d", this->vector.at(i));

			(i == this->vector.size() - 1) ? printf("|\n") : printf("|,");
		}
	}
};


class Player
{
public:
	int position;

	Player(int p = 1)
	{
		this->position = p;
	}

	//if true, player won the game
	bool move(int counter, const GameBoard *board)
	{
		this->position += counter;

		if (this->position >= (int)board->vector.size())
			return true;

		//trap
		this->position += board->vector.at(this->position);

		if (this->position < 0)
			this->position = 0;

		return false;
	}

	void resetPosition()
	{
		this->position = 1;
	}

	static int diceThrow()
	{
		return ((rand() % WALLS_ON_DICE) + 1);
	}
};


class X
{
public:
	int positionX1;
	int positionX2;
	int actualPlayerMove;

	X(int p1 = 0, int p2 = 0, int apm = 1)
	{
		this->positionX1 = p1;
		this->positionX2 = p2;
		this->actualPlayerMove = apm;
	}

	void display()
	{
		printf("(%d, %d, %d) ", this->positionX1, this->positionX2, this->actualPlayerMove);
	}

	void operator = (const X &other)
	{
		this->positionX1 = other.positionX1;
		this->positionX2 = other.positionX2;
		this->actualPlayerMove = other.actualPlayerMove;
	}

	bool operator == (const X &other)
	{
		if (this->positionX1 == other.positionX1 && this->positionX2 == other.positionX2 && this->actualPlayerMove == other.actualPlayerMove)
			return true;
		return false;
	}
};


class Equation
{
public:
	X myX;
	std::vector<X> unknowns;
	std::vector<int> numerators;
	int rest;

	Equation(const X &x)
	{
		this->myX = x;
		this->rest = 0;
	}

	void addX(const X &newX, const int boardSize)
	{
		if (newX.positionX1 >= boardSize || newX.positionX2 >= boardSize)
		{
			if (newX.actualPlayerMove == 2)
				this->rest++;
			return;
		}

		bool alreadyInEquation = false;
		for (int i = 0; i < (int)this->unknowns.size(); ++i)
		{
			if (this->unknowns.at(i) == newX)
			{
				alreadyInEquation = true;
				++(this->numerators.at(i));
				break;
			}
		}

		if (!alreadyInEquation)
		{
			this->unknowns.push_back(newX);
			this->numerators.push_back(1);
		}
	}

	void display()
	{
		this->myX.display();
		printf("= ");
		for (int i = 0; i < (int)this->unknowns.size(); ++i)
		{
			printf("%d/6*", this->numerators.at(i));
			this->unknowns.at(i).display();
			printf("+ ");
		}
		printf("%d/6\n", this->rest);
	}
};


class Game
{
public:
	GameBoard *board;
	std::vector<Player*> players;
	std::vector<Equation*> equations;

	Game()
	{
		srand((unsigned int)time(NULL));
		if (this->loadInput())
		{
			this->board->display();

			for (int i = 0; i < PLAYERS_NUMBER; ++i)
				this->players.push_back(new Player());

			monteCarloAlgorithm();

			this->generateAllXesInEquations();
			this->generateAllEquations();
			//this->displayAllEquations();

			this->gauss();
		}
	}

	~Game()
	{
		delete board;
		for (int i = this->players.size() - 1; i >= 0; --i)
			delete this->players[i];
		for (int i = this->equations.size() - 1; i >= 0; --i)
			delete this->equations[i];
	}

	void generateAllEquations()
	{
		int temp;
		for (int i = 0; i < (int)this->equations.size(); ++i)
		{
			for (int j = 1; j <= WALLS_ON_DICE; ++j)
			{
				if (this->equations.at(i)->myX.actualPlayerMove == 1)
				{
					temp = this->equations.at(i)->myX.positionX1 + j;
					if (temp < (int)this->board->vector.size())
						temp += this->board->vector.at(temp);
					this->equations.at(i)->addX(X(temp, this->equations.at(i)->myX.positionX2, 2), (int)this->board->vector.size());
				}
				else
				{
					temp = this->equations.at(i)->myX.positionX2 + j;
					if (temp < (int)this->board->vector.size())
						temp += this->board->vector.at(temp);
					this->equations.at(i)->addX(X(this->equations.at(i)->myX.positionX1, temp, 1), (int)this->board->vector.size());
				}
			}
		}
	}

	void gauss()
	{
		std::vector<std::vector<double>> matrixAB;
		std::vector<double> results(this->equations.size(), 0.0);

		this->createMatrixAB(matrixAB);

		this->gauss_Seidel(matrixAB);
		this->gaussElimination(matrixAB);
		this->calculateResults(matrixAB, results);
		printf("Gauss method: x(0, 0, 1) = \t%.15f\n", results.at(0));
	}

	void gauss_Seidel(std::vector<std::vector<double>> &AB)
	{
		const int size = (int)AB.size();
		int i, j, k;

		std::vector<double> results(size, 0);
		//Macierze: U - górnotrójk¹tna, L - dolnotrójk¹tna , D - diagonalna; 
		std::vector<std::vector<double>> U(size, results), L(size, results), D(size, results);

		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				if (i < j)
					U[i][j] = AB[i][j];
				else if (i > j)
					L[i][j] = AB[i][j];
				else
					D[i][j] = AB[i][j];
			}
		}

		// Calculate D^-1
		for (i = 0; i < size; i++)
			D[i][i] = 1 / D[i][i];

		// Calculate D^-1 * b
		for (i = 0; i < size; i++)
			AB[i][size] *= D[i][i];

		//Calculate D^-1 * L
		for (i = 0; i < size; i++)
			for (j = 0; j < i; j++)
				L[i][j] *= D[i][i];

		//Calculate D^-1 * U
		for (i = 0; i < size; i++)
			for (j = i + 1; j < size; j++)
				U[i][j] *= D[i][i];

		for (k = 0; k < MAX_GAUSS_SEIDELL; k++)
		{
			for (i = 0; i < size; i++)
			{
				results[i] = AB[i][size];
				for (j = 0; j < i; j++)
					results[i] -= L[i][j] * results[j];
				for (j = i + 1; j < size; j++)
					results[i] -= U[i][j] * results[j];
			}
			if (k != 0 && k % 10 == 0)
				displayGauss_Seidell(k, results.at(0));
		}
	}

	void displayGauss_Seidell(int i, double proba)
	{
		printf("Gauss Seidell: Iterations: %d, x(0, 0, 1) = \t%.15f\n", i, proba);
	}

	void displayMatrix(const std::vector<std::vector<double>> &matrixAB)
	{
		for (int i = 0; i < (int)this->equations.size(); ++i)
		{
			for (int j = 0; j < (int)this->equations.size() + 1; ++j)
			{
				printf("%.1f ", matrixAB.at(i).at(j));
			}
			printf("\n");
		}
	}

	void gaussElimination(std::vector<std::vector<double>> &matrixAB)
	{
		double coefficient = 0;
		//wybór kolumny
		for (int i = 0; i < (int)this->equations.size() - 1; ++i)
		{
			//wybór wiersza
			for (int j = i + 1; j < (int)this->equations.size(); ++j)
			{
				if (matrixAB.at(j).at(i) != 0.0)
				{
					coefficient = -matrixAB.at(j).at(i) / matrixAB.at(i).at(i);
					matrixAB.at(j).at(i) = 0.0; // -_-
					for (int k = i + 1; k <= (int)this->equations.size(); ++k)
					{
						matrixAB.at(j).at(k) += (coefficient * matrixAB.at(i).at(k));
					}
				}
			}
		}
	}

	void calculateResults(const std::vector<std::vector<double>> &matrixAB, std::vector<double> &results)
	{
		double productsSum = 0;
		for (int i = (int)results.size() - 1; i >= 0; --i)
		{
			productsSum = matrixAB.at(i).at((results.size()));
			for (int j = (int)results.size() - 1; j >= i + 1; --j)
			{
				if (results.at(j) != 0.0)
					productsSum -= matrixAB.at(i).at(j) * results.at(j);
			}
			results.at(i) = productsSum / matrixAB.at(i).at(i);
		}
	}

	void createMatrixAB(std::vector<std::vector<double>> &matrixAB)
	{
		for (int i = 0; i < (int)this->equations.size(); ++i)
		{
			std::vector<double> temp;
			for (int j = 0; j < (int)this->equations.size(); ++j)
			{
				temp.push_back(0.0);
				if (i == j)
					temp.at(i) = 1.0;
			}
			matrixAB.push_back(temp);
		}
		double temp;
		for (int i = 0; i < (int)this->equations.size(); ++i)
		{
			for (int j = 0; j < (int)this->equations.at(i)->unknowns.size(); ++j)
			{
				temp = (double)(this->equations.at(i)->numerators.at(j) / 6.0);
				//temp = round(temp);
				matrixAB.at(i).at(getXPlace(this->equations.at(i)->unknowns.at(j))) = temp;
			}
		}
		for (int i = 0; i < (int)this->equations.size(); ++i)
			matrixAB.at(i).push_back((double)(this->equations.at(i)->rest / 6.0));
	}

	int getXPlace(const X &other)
	{
		for (int i = 0; i < (int)this->equations.size(); ++i)
		{
			if (this->equations.at(i)->myX == other)
				return i;
		}
		return -1;
	}

	double round(const double &a)
	{
		double result = a;
		result *= 100;

		if (result < 0)
			result = ceil(result - 0.5);
		else
			result = floor(result + 0.5);

		return result / 100;
	}

	void generateAllXesInEquations()
	{
		//first equation x(0, 0, 1)
		this->equations.push_back(new Equation(X()));

		//adding rest equations for 1st player
		for (int i = 1; i < (int)this->board->vector.size(); ++i)
		{
			for (int j = 1; j < (int)this->board->vector.size(); ++j)
			{
				//no trap in fields => adding equation
				if (this->board->vector.at(i) == 0 && this->board->vector.at(j) == 0)
				{
					this->equations.push_back(new Equation(X(i, j, 1)));
				}
			}
		}

		for (int i = 1; i < (int)this->board->vector.size(); ++i)
		{
			//no trap in field => adding equation
			if (this->board->vector.at(i) == 0)
			{
				this->equations.push_back(new Equation(X(i, 0, 2)));
			}
		}

		//adding rest equations for 2nd player
		for (int i = 1; i < (int)this->board->vector.size(); ++i)
		{
			for (int j = 1; j < (int)this->board->vector.size(); ++j)
			{
				//no trap in fields => adding equation
				if (this->board->vector.at(i) == 0 && this->board->vector.at(j) == 0)
				{
					this->equations.push_back(new Equation(X(i, j, 2)));
				}
			}
		}
	}

	void displayAllEquations()
	{
		printf("Number of equations: %d\n", this->equations.size());
		for (int i = 0; i < (int)this->equations.size(); ++i)
		{
			printf("%d: ", i);
			this->equations.at(i)->display();
		}
	}

	//returns index of the player who won
	//if noone then -1
	int turnSimulation()
	{
		for (int i = 0; i < (int)this->players.size(); ++i)
		{
			if (this->players[i]->move(Player::diceThrow(), this->board))
			{
				return i;
			}
		}
		return -1;
	}

	//returns index of the player who won
	int gameSimulation()
	{
		int index = -1;
		while (true)
		{
			index = this->turnSimulation();
			if (index != -1)
			{
				this->resetPlayers();
				return index;
			}
		}
	}

	void monteCarloAlgorithm()
	{
		double probability = 0;
		int winner = -1;
		int counter = 0;
		bool quit = false;

		std::vector<double> results(5, 0);
		for (int i = 0; i < 100; ++i)
		{
			probability = 0.0;
			counter = 0;
			while (!quit)
			{
				counter++;
				winner = this->gameSimulation();
				if (winner == 0)
					probability += 1.0;

				switch (counter)
				{
				case 100:
					//displayProbability(probability, 1000);
					if (results[0] == 0)
						results[0] = probability;
					else
						results[0] = (results[0] + probability) / 2.0;
					break;
				case 1000:
					//displayProbability(probability, 10000);
					if (results[1] == 0)
						results[1] = probability;
					else
						results[1] = (results[1] + probability) / 2.0;
					break;
				case 10000:
					//displayProbability(probability, 100000);
					if (results[2] == 0)
						results[2] = probability;
					else
						results[2] = (results[2] + probability) / 2.0;
					break;
				case 100000:
					//displayProbability(probability, MAX_LOTTERY_GAMES);
					if (results[3] == 0)
						results[3] = probability;
					else
						results[3] = (results[3] + probability) / 2.0;
					break;
				case MAX_LOTTERY_GAMES:
					//displayProbability(probability, MAX_LOTTERY_GAMES);
					if (results[4] == 0)
						results[4] = probability;
					else
						results[4] = (results[4] + probability) / 2.0;
					quit = true;
					break;
				default:
					break;
				}
			}
			quit = false;
		}
		displayProbability(results[0], 100);
		displayProbability(results[1], 1000);
		displayProbability(results[2], 10000);
		displayProbability(results[3], 100000);
		displayProbability(results[4], MAX_LOTTERY_GAMES);
	}

	void displayProbability(double result, int counter)
	{
		printf("Monte Carlo for %d iterations:\t %.15f\n", counter, (result / (double)counter));
	}

	void resetPlayers()
	{
		for (int i = 0; i < (int)this->players.size(); ++i)
			this->players[i]->resetPosition();
	}

	bool loadInput()
	{
		FILE *inputFile;
		int boardSize, index, trapValue;

		fopen_s(&inputFile, "input.in", "r");

		if (inputFile)
		{
			fscanf_s(inputFile, "%d", &boardSize);
			board = new GameBoard(boardSize);

			while (fscanf_s(inputFile, ", {%d, %d}", &index, &trapValue))
			{
				board->vector.at(index) = trapValue;
			}
			return true;
		}
		return false;
	}
};


int main(int argc, char **argv)
{
	Game *game = new Game();

	delete game;
	return 0;
}