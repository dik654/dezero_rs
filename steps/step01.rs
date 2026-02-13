struct Variable {
    data: f64,
}

impl Variable {
    fn new(data: f64) -> Self {
        Variable { data }
    }
}

fn main() {
    let x = Variable::new(1.0);
    println!("x.data = {}", x.data);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable() {
        let x = Variable::new(1.0);
        assert_eq!(x.data, 1.0);
    }
}
