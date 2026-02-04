# Lab: Production Deployment

Deploy GenAI systems with guardrails and monitoring.

## Objectives

- Implement input/output guardrails
- Configure rate limiting
- Track production metrics

## Demo Code

See [`demos/course4/week6/production/`](https://github.com/noahgift/DB-mlops-genai/tree/main/demos/course4/week6/production)

## Lab Exercise

See [`labs/course4/week6/lab_6_3_production.py`](https://github.com/noahgift/DB-mlops-genai/tree/main/labs/course4/week6)

## Key Implementation

```rust
pub struct ProductionServer {
    guardrails: Guardrails,
    rate_limiter: RateLimiter,
    metrics: Metrics,
    router: ABRouter,
}

impl ProductionServer {
    pub fn process(&mut self, request: Request) -> Response {
        // 1. Check rate limit
        if !self.rate_limiter.check() {
            return Response::error("Rate limited");
        }

        // 2. Check guardrails
        let check = self.guardrails.check_input(&request.prompt);
        if !check.passed {
            self.metrics.record_guardrail_block();
            return Response::error("Blocked by guardrails");
        }

        // 3. Route to model variant
        let model = self.router.select();

        // 4. Generate and record metrics
        let start = Instant::now();
        let response = model.generate(&request.prompt);
        self.metrics.record(start.elapsed(), response.tokens);

        response
    }
}
```
