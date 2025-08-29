use owo_colors::OwoColorize;
use std::collections::HashSet;
use std::fmt;

#[derive(Debug, Clone, Copy)]
struct VarId(usize);

#[derive(Debug, Clone, Copy)]
struct RobddId(usize);

#[derive(Debug, Clone, Copy)]
enum RobddNodeKind {
    Node {
        var: RobddId,
        high: RobddId,
        low: RobddId,
    },
    Leaf(bool),
}
#[derive(Debug, Clone, Copy)]
struct RobddNode {
    kind: RobddNodeKind,
    hash: u64,
}

#[derive(Debug, Clone)]

struct Robdd {
    nodes: Vec<RobddNode>,
}

// contains the truth table of the boolean function.
// Each entry in the set is a vector of variable assignments that yield true.
#[derive(Debug, Clone)]
struct Factor {
    // the truth table of the boolean function.
    bool_fn: TruthTable,
    // the coefficient of the factor.
    coeff: i64,
}

#[derive(Debug, Clone)]
struct Term {
    factors: Vec<Factor>,
}

#[derive(Debug, Clone)]
struct Literal {
    var: usize, // VarId assumed usize for demo
    negated: bool,
}

#[derive(Debug, Clone)]
struct ConjClause {
    vars: Vec<Literal>,
}

#[derive(Debug, Clone)]
struct Dnf {
    clauses: Vec<ConjClause>,
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.negated {
            write!(f, "¬x{}", self.var)
        } else {
            write!(f, "x{}", self.var)
        }
    }
}

impl fmt::Display for ConjClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.vars.iter().map(|lit| lit.to_string()).collect();
        write!(f, "({})", parts.join("&"))
    }
}

impl fmt::Display for Dnf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.clauses.iter().map(|cl| cl.to_string()).collect();
        if parts.is_empty() {
            return write!(f, "(F)");
        }
        write!(f, "{}", parts.join("|"))
    }
}

// Environment to hold variable assignments
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Env<T> {
    assigns: Vec<T>,
}

impl<T: Clone> Env<T> {
    fn new(assigns: Vec<T>) -> Self {
        Env { assigns }
    }

    fn get(&self, var: &VarId) -> T {
        self.assigns[var.0].clone()
    }
}

type BoolEnv = Env<bool>;
type IntEnv = Env<i64>;

impl BoolEnv {
    fn to_clause(&self) -> ConjClause {
        let mut vars = Vec::new();
        for (i, &val) in self.assigns.iter().enumerate() {
            vars.push(Literal {
                var: i,
                negated: !val,
            });
        }
        ConjClause { vars }
    }
}

impl IntEnv {
    fn to_bool_env_by_slice_bits(&self, slice: usize) -> BoolEnv {
        let mut assigns = Vec::new();
        for i in 0..self.assigns.len() {
            assigns.push((self.assigns[i] & (1 << slice)) == 1);
        }
        BoolEnv::new(assigns)
    }

    // fn of_bool_env(env: &BoolEnv, slice : usize) -> Self {
    //     let mut assigns = Vec::new();
    //     for i in 0..env.assigns.len() {
    //         assigns.push((env.assigns[i] as i64) << slice);
    //     }
    //     IntEnv::new(assigns)
    // }
}

impl fmt::Display for IntEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // write x_i = v_i
        write!(f, "<")?;
        for (i, &val) in self.assigns.iter().enumerate() {
            write!(f, "x{}:{}", i, val)?;
            if i + 1 < self.assigns.len() {
                write!(f, ", ")?;
            }
        }
        write!(f, ">")?;
        Ok(())
    }
}

// impl BoolEnv {
//     fn to_int_env(&self) -> IntEnv {
//         IntEnv::of_bool_env(self, 0)
//     }
//
// }

#[derive(PartialEq, Eq, Debug, Clone)]
struct TruthTable {
    // the truth table, which contains all variable assignments that yield true.
    table: HashSet<BoolEnv>,
}

// generate all truth tables for n variables
fn truth_tables(nvars: usize) -> Vec<TruthTable> {
    let mut tables = Vec::new();
    let total_entries = 1 << nvars; // 2^nvars
    for i in 0..(1 << total_entries) {
        // 2^(2^nvars)
        let mut bool_fn = TruthTable::new();
        for j in 0..total_entries {
            if (i & (1 << j)) != 0 {
                let assigns: Vec<bool> = (0..nvars).map(|k| (j & (1 << k)) != 0).collect();
                bool_fn.add_true(Env { assigns });
            }
        }
        tables.push(bool_fn);
    }
    tables
}

impl TruthTable {
    fn new() -> Self {
        TruthTable {
            table: HashSet::new(),
        }
    }

    fn add_true(&mut self, entry: BoolEnv) {
        self.table.insert(entry);
    }

    fn eval_bool(&self, env: &BoolEnv) -> bool {
        self.table.contains(env)
    }

    fn eval_int_bitwise(&self, env: &IntEnv) -> i64 {
        // iterate from [0, 64), take slices of 'BoolEnv',
        // evaluate and shift.
        // eval bitwise
        let mut result = 0;
        for i in 0..64 {
            let slice = env.to_bool_env_by_slice_bits(i);
            result |= (self.eval_bool(&slice) as i64) << i;
        }
        result
    }

    fn to_dnf(&self) -> Dnf {
        Dnf {
            clauses: self.table.iter().map(|env| env.to_clause()).collect(),
        }
    }
}

impl fmt::Display for TruthTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_dnf().fmt(f)
    }
}

impl Factor {
    fn new(bool_fn: TruthTable, coeff: i64) -> Self {
        Factor { bool_fn, coeff }
    }

    fn eval_int(&self, env: &IntEnv) -> i64 {
        self.coeff * self.bool_fn.eval_int_bitwise(env)
    }
}

impl fmt::Display for Factor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // print +/- for coeff
        let sign = if self.coeff < 0 { "-" } else { "+" };
        write!(f, "{}{}{}", sign, self.coeff.abs(), self.bool_fn)
    }
}

impl Term {
    fn eval_int(&self, env: &IntEnv) -> i64 {
        let mut result = 0;
        for factor in &self.factors {
            let factor_value = factor.eval_int(env);
            result += factor_value;
        }
        result
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // write separated by space.
        let mut entries: Vec<String> = self
            .factors
            .iter()
            .map(|factor| factor.to_string())
            .collect();
        entries.sort();
        if entries.is_empty() {
            entries.push("0".to_string());
        }
        write!(f, "{}", entries.join(" "))
    }
}

// a Generatable<T> allows us to get first() and next() values of type T.

struct Generator {
    bool_fns: Vec<TruthTable>,
    nvars: usize,
    maxcoeff: usize,   // max coefficient value.
    maxfactors: usize, // max number of factors.
    max_check_upto: usize,
}

impl Generator {
    fn first_truth_table(&self) -> Option<&TruthTable> {
        self.bool_fns.first()
    }

    fn next_truth_table(&self, bool_fn: &TruthTable) -> Option<TruthTable> {
        let idx = self.bool_fns.iter().position(|x| x == bool_fn)?;
        if idx + 1 < self.bool_fns.len() {
            Some(self.bool_fns[idx + 1].clone())
        } else {
            None
        }
    }

    fn first_coeff(&self) -> Option<i64> {
        Some(-(self.maxcoeff as i64))
    }

    fn next_coeff(&self, coeff: i64) -> Option<i64> {
        if coeff < self.maxcoeff as i64 {
            Some(coeff + 1)
        } else {
            None
        }
    }

    fn first_factor(&self) -> Option<Factor> {
        let bool_fn = self.first_truth_table()?.clone();
        let coeff = self.first_coeff()?;
        Some(Factor::new(bool_fn, coeff))
    }

    fn next_factor(&self, factor: &Factor) -> Option<Factor> {
        match self.next_coeff(factor.coeff) {
            Some(next_coeff) => Some(Factor::new(factor.bool_fn.clone(), next_coeff)),
            None => match self.next_truth_table(&factor.bool_fn) {
                Some(next_bool_fn) => Some(Factor::new(next_bool_fn, self.first_coeff()?)),
                None => None,
            },
        }
    }

    fn first_term(&self) -> Option<Term> {
        Some(Term { factors: vec![] })
    }

    fn next_term(&self, t: &Term) -> Option<Term> {
        let mut new_term = t.clone();
        for i in (0..new_term.factors.len()).rev() {
            if let Some(next_factor) = self.next_factor(&new_term.factors[i]) {
                new_term.factors[i] = next_factor;
                return Some(new_term);
            }
        }
        if new_term.factors.len() < self.maxfactors {
            if let Some(first_factor) = self.first_factor() {
                new_term.factors.push(first_factor);
                return Some(new_term);
            }
        }
        None
    }
}

impl Term {
    fn is_identically_zero(&self, envs: IntEnvIter) -> Option<IntEnv> {
        for env in envs {
            if self.eval_int(&env) != 0 {
                return Some(env);
            }
        }
        None
    }
}

struct IntEnvIter {
    nvars: usize,
    min_val: i64,
    max_val: i64,
    current: Vec<i64>,
    done: bool,
}

impl IntEnvIter {
    fn new(nvars: usize, min_val: i64, max_val: i64) -> Self {
        IntEnvIter {
            nvars,
            min_val,
            max_val,
            done: false,
            current: vec![min_val; nvars],
        }
    }

    fn new_bool(nvars: usize) -> Self {
        IntEnvIter::new(nvars, 0, 1)
    }
}

// return if we could increment
fn env_next(env: &mut Vec<i64>, min_val: i64, max_val: i64) -> bool {
    for i in 0..env.len() {
        if env[i] < max_val {
            env[i] += 1;
            for j in 0..i {
                env[j] = min_val;
            }
            return true;
        }
    }
    false
}

impl Iterator for IntEnvIter {
    type Item = Env<i64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // produce current environment
        let out = Env {
            assigns: self.current.clone(),
        };

        // increment like an odometro (odometer)
        let did_increment = env_next(&mut self.current, self.min_val, self.max_val);
        self.done = !did_increment;
        Some(out)
    }
}

fn print_term_eval_table(t: &Term, iter: IntEnvIter) {
    for env in iter {
        let val = t.eval_int(&env);
        let str = format!("   @ {} => {}", env, val);
        let str_colored = if val == 0 {
            str.dimmed().to_string()
        } else {
            str
        };
        println!("{}", str_colored);
    }
}

enum CheckTermResult {
    Good,
    Skip(IntEnv),
    Error(IntEnv),
}

fn check_term(g: &Generator, t: &Term) -> CheckTermResult {
    // check if eqn holds with inputs {0, 1}, and then check if it
    // holds for inputs being bitvectors where the relations hold.
    if let Some(cex) = t.is_identically_zero(IntEnvIter::new_bool(g.nvars)) {
        print!("{}:", "SKIP".yellow());
        println!(
            "Term {} evaluates to nonzero value {} @ boolenv {}",
            t,
            t.eval_int(&cex),
            cex
        );
        // print_term_eval_table(t, IntEnvIter::new_bool(g.nvars));
        return CheckTermResult::Skip(cex);
    }

    // because equation holds, check on larger outputs.
    if let Some(cex) = t.is_identically_zero(IntEnvIter::new(
        g.nvars,
        -(g.max_check_upto as i64),
        g.max_check_upto as i64,
    )) {
        print!("{}:", "ERROR".red());
        println!(
            "Term {} that was identically zero on {{0,1}} evaluates to nonzero value {:?} @ env {}",
            t,
            t.eval_int(&cex),
            cex
        );

        print_term_eval_table(t, IntEnvIter::new_bool(g.nvars));
        CheckTermResult::Error(cex)
    } else {
        print!("{}:", "GOOD".green());
        println!(
            "Term {} is identically zero both on booleans and bitvectors",
            t
        );
        CheckTermResult::Good
    }
}

fn main() {
    let g = Generator {
        bool_fns: truth_tables(2),
        nvars: 2,
        maxcoeff: 4,
        maxfactors: 5,
        max_check_upto: 3,
    };

    let mut t = g
        .first_term()
        .unwrap_or_else(|| panic!("No first term available"));

    let mut errors = Vec::new();
    let mut good = Vec::new();
    let mut skips = Vec::new();
    loop {
        let check_result = check_term(&g, &t);
        match check_result {
            CheckTermResult::Good => good.push(t.clone()),
            CheckTermResult::Skip(env) => skips.push((t.clone(), env)),
            CheckTermResult::Error(env) => {
                errors.push((t.clone(), env));
                break;
            }
        }

        // get the next term.
        if let Some(next_term) = g.next_term(&t) {
            t = next_term;
        } else {
            break;
        }
    }

    println!("\nSummary:");
    println!(
        "  {} terms checked",
        good.len() + skips.len() + errors.len()
    );
    println!("  {} terms are identically zero", good.len());
    println!("  {} terms skipped", skips.len());
    println!("  {} terms found with errors", errors.len());

    for (ter, env) in errors.iter() {
        println!("Error term: {} @ env {} = {}", ter, env, ter.eval_int(env));
    }

    // set exit code based on errors
    if !errors.is_empty() {
        std::process::exit(1);
    } else {
        std::process::exit(0);
    }
}
