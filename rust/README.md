## Testing

To test the basic ollama integration, run in root:

```
cargo test --features ollama
```

To see debug output add:

```
cargo test --features ollama -- --nocapture
```

## TODO

- [x] Fix Ollama adapter tests
- [ ] OpenAI adapter + tests
- [ ] Anthropic adapter + tests

## Notes

UnresolvedResponse could use method chaining like ContextBuilder but use cases are pretty niche
