import pickle

with open('data/test_out/holstep000', 'rb') as f:
    data = pickle.load(f)

print(f'Total records: {len(data)}')

for i, (flag, conj, stmt) in enumerate(data[:3]):
    print(f'Record {i}: flag={flag}')
    print(f'Conjecture type: {type(conj)}')
    print(f'Statement type: {type(stmt)}')
