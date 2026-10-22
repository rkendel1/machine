//! The agent's one tool: an exact integer calculator (+ - * / and parentheses).
//! Division must be exact so the tool never silently rounds.

use anyhow::{Result, bail};

pub fn eval(src: &str) -> Result<i64> {
    let toks = lex(src)?;
    let mut p = Parser { toks: &toks, i: 0 };
    let v = p.expr()?;
    if p.i != toks.len() {
        bail!("unexpected {:?}", toks[p.i]);
    }
    Ok(v)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tok {
    Num(i64),
    Op(char),
}

fn lex(s: &str) -> Result<Vec<Tok>> {
    let mut out = Vec::new();
    let cs: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() {
            let mut n: i64 = 0;
            // Accept thousands separators: 1,200 and 1_200.
            let sep = |i: usize| {
                (cs[i] == ',' || cs[i] == '_') && cs.get(i + 1).is_some_and(|d| d.is_ascii_digit())
            };
            while i < cs.len() && (cs[i].is_ascii_digit() || sep(i)) {
                if cs[i].is_ascii_digit() {
                    n = n
                        .checked_mul(10)
                        .and_then(|n| n.checked_add(cs[i].to_digit(10).unwrap() as i64))
                        .ok_or_else(|| anyhow::anyhow!("number too large"))?;
                }
                i += 1;
            }
            out.push(Tok::Num(n));
        } else if "+-*/()x×".contains(c) {
            out.push(Tok::Op(match c {
                'x' | '×' => '*',
                c => c,
            }));
            i += 1;
        } else {
            bail!("unsupported character {c:?}");
        }
    }
    if out.is_empty() {
        bail!("empty expression");
    }
    Ok(out)
}

struct Parser<'a> {
    toks: &'a [Tok],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<Tok> {
        self.toks.get(self.i).copied()
    }
    fn expr(&mut self) -> Result<i64> {
        let mut v = self.term()?;
        while let Some(Tok::Op(op @ ('+' | '-'))) = self.peek() {
            self.i += 1;
            let r = self.term()?;
            v = if op == '+' {
                v.checked_add(r)
            } else {
                v.checked_sub(r)
            }
            .ok_or_else(|| anyhow::anyhow!("overflow"))?;
        }
        Ok(v)
    }
    fn term(&mut self) -> Result<i64> {
        let mut v = self.factor()?;
        while let Some(Tok::Op(op @ ('*' | '/'))) = self.peek() {
            self.i += 1;
            let r = self.factor()?;
            v = if op == '*' {
                v.checked_mul(r)
                    .ok_or_else(|| anyhow::anyhow!("overflow"))?
            } else {
                if r == 0 {
                    bail!("division by zero");
                }
                if v % r != 0 {
                    bail!("{v}/{r} is not an integer");
                }
                v / r
            };
        }
        Ok(v)
    }
    fn factor(&mut self) -> Result<i64> {
        match self.peek() {
            Some(Tok::Num(n)) => {
                self.i += 1;
                Ok(n)
            }
            Some(Tok::Op('-')) => {
                self.i += 1;
                Ok(-self.factor()?)
            }
            Some(Tok::Op('(')) => {
                self.i += 1;
                let v = self.expr()?;
                if self.peek() != Some(Tok::Op(')')) {
                    bail!("missing ')'");
                }
                self.i += 1;
                Ok(v)
            }
            t => bail!("expected a number, got {t:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::eval;

    #[test]
    fn arithmetic() {
        assert_eq!(eval("2+3*4").unwrap(), 14);
        assert_eq!(eval("(2+3)*4").unwrap(), 20);
        assert_eq!(eval("-3 + 10 / 2").unwrap(), 2);
        assert_eq!(eval("12 x 7").unwrap(), 84);
        assert_eq!(eval("1,200 + 1").unwrap(), 1201);
        assert!(eval("7/2").is_err());
        assert!(eval("2+").is_err());
        assert!(eval("abc").is_err());
    }
}
