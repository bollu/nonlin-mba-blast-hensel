#include <cvc5/cvc5.h>
#include <iostream>

int main() {
    cvc5::Solver solver;
    solver.setLogic("QF_LIA");
    auto x = solver.mkConst(solver.getIntegerSort(), "x");
    auto zero = solver.mkInteger(0);
    auto gt = solver.mkTerm(cvc5::Kind::GT, {x, zero});
    solver.assertFormula(gt);
    std::cout << "Check sat: " << solver.checkSat() << std::endl;
}

