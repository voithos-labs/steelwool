# `steelwool 🧶🔗`

lightweight library for interacting with LLMs

## TODO

- [x] Core impl
- [ ] Fully generalized stream handling
- [ ] Unified tool format
- [ ] Node (typescript) & python port

## Testing

To test the basic ollama integration, run in root:

```
cargo test --features ollama
```

To see debug output add:

```
cargo test --features ollama -- --nocapture
```
