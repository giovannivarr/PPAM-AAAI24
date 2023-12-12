echo "------------------------------------------------"
echo "Generating Q-learning variant with shield"
python watertank.py --num-steps $1 -o 1 -t $2 -c ./data/watertank_shield_q --seed ${3:-0}

echo "------------------------------------------------"
echo "Generating Q-learning variant with PPAM"
python watertank.py --num-steps $1 -m -t $2 -c ./data/watertank_ppam_q --seed ${3:-0}

echo "------------------------------------------------"
echo "Generating SARSA variant with shield"
python watertank.py --num-steps $1 -o 1 -t $2 -r -c ./data/watertank_shield_sarsa --seed ${3:-0}

echo "------------------------------------------------"
echo "Generating SARSA variant with PPAM"
python watertank.py --num-steps $1 -m -t $2 -r -c ./data/watertank_ppam_sarsa --seed ${3:-0}