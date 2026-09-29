chcp 65001
::test_llm.exe -m gemma3-4b.gguf --threads 4 -c 2048 --lang ru --temp 0.6 -p "расскажи мне сказку"
test_llm.exe -m cpm5-2b.gguf --threads 4 -c 2048 --lang ru --temp 0.6 -p "расскажи мне сказку"
