use std::collections::HashSet;

#[derive(Debug, Clone, Copy)]
enum Binop {
    And,
    Or,
    Xor,
}

struct VarId(usize);

struct RobddId(usize);
enum RobddNodeKind {
    Node {
        var: RobddId,
        high: RobddId,
        low: RobddId,
    },
    Leaf(bool),
}

struct RobddNode {
    kind: RobddNodeKind,
    hash: u64,
}

struct Robdd {
    nodes : Vec<RobddNode>
}

// contains the truth table of the boolean function.
// Each entry in the set is a vector of variable assignments that yield true.
#[derive(Debug, Clone)]
struct Factor {
    // the truth table of the boolean function.
    tt : TruthTable,
    // the coefficient of the factor.
    coeff : i64,
}

#[derive(Debug, Clone)]
struct Term {
    factors : Vec<Factor>,
}

// Environment to hold variable assignments
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Env<T> {
    assigns : Vec<T>,
}

impl<T : Clone> Env<T> {
    fn new(assigns: Vec<T>) -> Self {
        Env { assigns }
    }

    fn get(&self, var: &VarId) -> T {
        self.assigns[var.0].clone()
    }
}

type BoolEnv = Env<bool>;
type IntEnv = Env<i64>;

impl IntEnv {
    fn to_bool_env(&self, slice : usize) -> BoolEnv {
        let mut assigns = Vec::new();
        for i in 0..self.assigns.len() {
            assigns.push((self.assigns[i] >> slice) & 1 == 1);
        }
        BoolEnv::new(assigns)
    }

    fn of_bool_env(env: &BoolEnv, slice : usize) -> Self {
        let mut assigns = Vec::new();
        for i in 0..env.assigns.len() {
            assigns.push((env.assigns[i] as i64) << i);
        }
        IntEnv::new(assigns)
    }
}

impl BoolEnv {
    fn to_int_env(&self) -> IntEnv {
        IntEnv::of_bool_env(&self, 0)
    }

}

#[derive(PartialEq, Eq, Debug, Clone)]
struct TruthTable {
    // the truth table, which contains all variable assignments that yield true.
    table : HashSet<BoolEnv>,
}


// generate all truth tables for n variables
fn truth_tables(nvars : usize) -> Vec<TruthTable> {
    let mut tables = Vec::new();
    let total_entries = 1 << nvars; // 2^nvars
    for i in 0..(1 << total_entries) { // 2^(2^nvars)
        let mut tt = TruthTable::new();
        for j in 0..total_entries {
            if (i & (1 << j)) != 0 {
                let assigns : Vec<bool> = (0..nvars).map(|k| (j & (1 << k)) != 0).collect();
                tt.add_true(Env { assigns });
            }
        }
        tables.push(tt);
    }
    tables
}

impl TruthTable {
    fn new() -> Self {
        TruthTable { table: HashSet::new() }
    }
    
    fn add_true(&mut self, entry: BoolEnv) {
        self.table.insert(entry);
    }

    fn eval_bool(&self, env: &BoolEnv) -> bool {
        self.table.contains(&env)
    }

    fn eval_int(&self, env: &IntEnv) -> i64 {
        // iterate from [0, 64), take slices of 'BoolEnv',
        // evaluate and shift.
        let mut result = 0;
        for i in 0..64 {
            let slice = env.to_bool_env(i);
            result |= (self.eval_bool(&slice) as i64) << i;
        }
        result
    }

}

impl Factor {
    fn new(tt: TruthTable, coeff: i64) -> Self {
        Factor { tt, coeff }
    }

    fn next_coeff(&self) -> Option<i64> {
        // Generate the next coefficient in some order (e.g., incrementing)
        // Placeholder implementation
        None
    }
}

impl Term {
    fn new() -> Self {
        Term { factors: Vec::new() }
    }

    fn add_factor(&mut self, factor: Factor) {
        self.factors.push(factor);
    }

    fn eval_int(&self, env: &IntEnv) -> i64 {
        let mut result = 0;
        for factor in &self.factors {
            let factor_value = factor.coeff * (factor.tt.eval_int(env) as i64);
            result += factor_value;
        }
        result
    }
}

// a Generatable<T> allows us to get first() and next() values of type T.

struct Generator {
    tts : Vec<TruthTable>,
    nvars : usize,
    maxcoeff : usize, // max coefficient value.
    maxfactors : usize, // max number of factors.
    maxvarval : usize
}

impl Generator {

    fn first_truth_table(&self) -> Option<&TruthTable> {
        self.tts.first()
    }

    fn next_truth_table(&self, tt : &TruthTable) -> Option<TruthTable> {
        let idx = self.tts.iter().position(|x| x == tt)?;
        if idx + 1 < self.tts.len() {
            Some(self.tts[idx + 1].clone())
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
        let tt = self.first_truth_table()?.clone();
        let coeff = self.first_coeff()?;
        Some(Factor::new(tt, coeff))
    }

    fn next_factor(&self, factor: &Factor) -> Option<Factor> {
        match self.next_coeff(factor.coeff) {
            Some(next_coeff) => Some(Factor::new(factor.tt.clone(), next_coeff)),
            None => match self.next_truth_table(&factor.tt) {
                Some(next_tt) => Some(Factor::new(next_tt, self.first_coeff()?)),
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
    fn is_identically_zero (&self, envs : IntEnvIter) -> Option<IntEnv> {
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
}

impl IntEnvIter {
    fn new(nvars: usize, min_val: i64, max_val: i64) -> Self {
        IntEnvIter {
            nvars,
            min_val,
            max_val,
            current: vec![min_val; nvars],
        }
    }

    fn new_bool(nvars : usize) -> Self {
        IntEnvIter::new(nvars, 0, 1)
    }
}


impl Iterator for IntEnvIter {
    type Item = Env<i64>;

    fn next(&mut self) -> Option<Self::Item> {

        // produce current environment
        let env = Env {
            assigns: self.current.clone(),
        };

        // increment like an odometro (odometer)
        for i in (0..self.nvars).rev() {
            if self.current[i] < self.max_val {
                self.current[i] += 1;
                for j in i+1..self.nvars {
                    self.current[j] = self.min_val;
                }
                return Some(env);
            }
        }

        // if we reach here, we are done
        None
    }
}

fn check_term(g : &Generator, t : &Term) {
    // check if eqn holds with inputs {0, 1}, and then check if it
    // holds for inputs being bitvectors where the relations hold.
    if let Some(cex) = t.is_identically_zero (IntEnvIter::new_bool(g.nvars)) {
        println!("BAD: Term {:?} evaluates to nonzero value {:?} @ env {:?}",
            t, t.eval_int(&cex), cex);
        return; 
    }

    // because equation holds, check on larger outputs. 
    if let Some(cex) = t.is_identically_zero (IntEnvIter::new(g.nvars,
        -(g.maxvarval as i64),
        g.maxvarval as i64)) {
        println!("ERROR: Term {:?} that was true on the booleans evaluates to nonzero value {:?} @ env {:?}",
            t, t.eval_int(&cex), cex);
        return; 
    } else {
        println!("GOOD: Term {:?} is identically zero both on booleans and bitvectors", t);
    }

}

fn main() {
    let g = Generator {
        tts: truth_tables(2),
        nvars: 2,
        maxcoeff: 3,
        maxfactors: 2,
        maxvarval: 2
    };

    let mut t = g.first_term().unwrap_or_else(||
        panic!("No first term available")
    );

    loop {
        check_term(&g, &t);
        // get the next term.
        if let Some(next_term) = g.next_term(&t) {
            t = next_term;
        } else {
            break;
        }
    }
}
