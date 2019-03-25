#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <assert.h>

constexpr const int BoardSize = 8;
constexpr const int BoardCells = BoardSize * BoardSize;

// static_assert (BoardSize % 2 == 0, "BoardSize should be even");
enum class CellStatus {
    None, Black, White, BlackQueen, WhiteQueen, Forbidden
};

enum class Team {
    None, Black, White
};

enum class Winner {
    White,
    Black,
    Draw,
    NotFinished
};

static Team TeamOfCell(CellStatus status){
    if (status == CellStatus::Black || status == CellStatus::BlackQueen) {
        return Team::Black;
    }
    if (status == CellStatus::White || status == CellStatus::WhiteQueen) {
        return Team::White;
    }
    return Team::None;
}

static bool Equal(Team team, CellStatus cell) {
    return (team == Team::Black && (cell == CellStatus::Black || cell == CellStatus::BlackQueen))||
            (team == Team::White && (cell == CellStatus::White || cell == CellStatus::WhiteQueen));
}

static bool operator==(Team team, CellStatus cell) {
    return Equal(team, cell);
}

static bool operator!=(Team team, CellStatus cell) {
    return !(team == cell);
}

static bool operator!=(CellStatus cell, Team team) {
    return !(team == cell);
}

static bool operator==(CellStatus cell, Team team) {
    return team == cell;
}

static Team Opponent(Team team) {
    assert(team != Team::None);
    if(team == Team::Black) {
        return Team::White;
    }
    return Team::Black;
}

static bool IsQueen(CellStatus cell) {
    if (cell == CellStatus::BlackQueen || cell == CellStatus::WhiteQueen) {
        return true;
    }
    assert(cell == CellStatus::Black || cell == CellStatus::White);
    return false;
}

struct GameState {
    std::array<CellStatus, BoardCells> State;
    const GameState* Parent = nullptr;
    Team CurrentTeam;


    GameState(Team team = Team::White) : CurrentTeam(team) {
        for (int i = 0; i < BoardSize; ++i) {
            for (int j = 0; j < BoardSize; ++j) {
                auto& pos = State[Index(i, j)];
                if (i < BoardSize / 2 - 1) {
                    if (j % 2 == i % 2) {
                        pos = CellStatus::Black;
                        continue;
                    }
                } else if (i > BoardSize / 2) {
                    if (j % 2 == i % 2) {
                        pos = CellStatus::White;
                        continue;
                    }
                }
                pos = CellStatus::None;
            }
        }
    }
    GameState(const GameState* parent) : State(parent->State), Parent(parent), CurrentTeam(Opponent(parent->CurrentTeam)) {
    }

    GameState(const GameState&) = default;
    GameState(GameState&& other) {
        std::swap(other.State, State);
        CurrentTeam = other.CurrentTeam;
    }

    static std::shared_ptr<GameState> CreateEmpty();

    bool Equal(GameState& other) const {
        return State == other.State;
    }

    std::size_t Hash() const {
        std::size_t val = 16;
        for(auto& v : State) {
            unsigned int i = static_cast<unsigned int>(typename std::underlying_type<CellStatus>::type(v));
            val ^= i + 0x9e3779b9 + (val << 6) + (val >> 2);
        }
        return val;
    }

    static unsigned int Index(int i, int j) {
        return static_cast<unsigned int>(i * BoardSize + j);
    }

    bool IsTerminal() const;
    Winner GetWinner() const {
        if (IsTerminal()) {
            return CurrentTeam == Team::White ? Winner::Black : Winner::White;
        }
        return Winner::NotFinished;
    }

    CellStatus At(int i, int j) const {
        if(i < 0 || i >= BoardSize || j < 0 || j >= BoardSize) {
            return CellStatus::Forbidden;
        }
        return State[Index(i, j)];
    }
    int RegularChecks(Team team);
    int QueenChecks(Team team);

    CellStatus& Cell(int i, int j);
    void SetCell(int i, int j, CellStatus status) {
        Cell(i, j) = status;
    }


    std::vector<std::shared_ptr<GameState>> KillingMovesFor(Team team) const;
    std::vector<std::shared_ptr<GameState>> NonKillingMovesFor(Team team) const;

    std::vector<std::shared_ptr<GameState> > KillingMovesForQueen(int i, int j) const;
    std::vector<std::shared_ptr<GameState> > KillingMovesForRegular(int i, int j) const;
    std::vector<std::shared_ptr<GameState> > NonKillingMovesForQueen(int i, int j) const;
    std::vector<std::shared_ptr<GameState> > NonKillingMovesForRegular(int i, int j) const;

    const std::vector<std::shared_ptr<GameState> >& Expand() const;

    void Dump(std::ostream& stream) const;

    bool operator==(GameState& other) const {
        return other.State == State && other.CurrentTeam == CurrentTeam;
    }

private:
    static const CellStatus forbidden = CellStatus::Forbidden;
    mutable std::vector<std::shared_ptr<GameState> > expansion;
    mutable bool cached_expansion_ = false;
};
