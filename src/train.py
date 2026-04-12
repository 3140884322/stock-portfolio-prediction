import random
import numpy as np
import os
import torch as torch
from load_data import load_EOD_data
from evaluator import evaluate
from model import get_loss, StockMixer
import pickle


np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

data_path = 'dataset'
market_name = 'NASDAQ'
relation_name = 'wikidata'
stock_num = 1026
lookback_length = 16
epochs = 100
valid_index = 756
test_index = 1008
fea_num = 5
market_num = 20
steps = 1
learning_rate = 0.001
alpha = 0.1
scale_factor = 3
activation = 'GELU'

dataset_path = '/dataset/' + market_name
if market_name == "SP500":
    data = np.load('../dataset/SP500/SP500.npy')
    data = data[:, 915:, :]
    price_data = data[:, :, -1]
    mask_data = np.ones((data.shape[0], data.shape[1]))
    eod_data = data
    gt_data = np.zeros((data.shape[0], data.shape[1]))
    for ticket in range(0, data.shape[0]):
        for row in range(1, data.shape[1]):
            gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / \
                                   data[ticket][row - steps][-1]
else:
    with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
        eod_data = pickle.load(f)
    with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
        mask_data = pickle.load(f)
    with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
        gt_data = pickle.load(f)
    with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
        price_data = pickle.load(f)

trade_dates = mask_data.shape[1]
model = StockMixer(
    stocks=stock_num,
    time_steps=lookback_length,
    channels=fea_num,
    market=market_num,
    scale=scale_factor
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_valid_loss = np.inf
best_valid_perf = None
best_test_perf = None
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

def portfolio_backtest(pred, gt, mask, top_k=5):
    """
    pred: [stock_num, num_days] predicted scores/returns
    gt:   [stock_num, num_days] actual next-period returns
    mask: [stock_num, num_days] valid mask
    """
    daily_returns = []

    num_days = pred.shape[1]

    for d in range(num_days):
        valid_idx = mask[:, d] > 0
        if np.sum(valid_idx) < top_k:
            continue

        day_pred = pred[valid_idx, d]
        day_gt = gt[valid_idx, d]

        top_idx = np.argsort(day_pred)[-top_k:]   # top K predicted stocks
        top_pred = day_pred[top_idx]
        top_gt = day_gt[top_idx]

        exp_scores = np.exp(top_pred - np.max(top_pred))
        weights = exp_scores / np.sum(exp_scores)

        portfolio_ret = np.sum(weights * top_gt)
        daily_returns.append(portfolio_ret)

    daily_returns = np.array(daily_returns)
    cumulative_returns = np.cumprod(1 + daily_returns) - 1

    sharpe = 0.0
    if np.std(daily_returns) > 1e-12:
        sharpe = np.mean(daily_returns) / np.std(daily_returns)

    return {
        "daily_returns": daily_returns,
        "cumulative_returns": cumulative_returns,
        "mean_daily_return": np.mean(daily_returns) if len(daily_returns) > 0 else 0.0,
        "final_cumulative_return": cumulative_returns[-1] if len(cumulative_returns) > 0 else 0.0,
        "sharpe": sharpe
    }

def validate(start_index, end_index):
    with torch.no_grad():
        cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(

                lambda x: torch.Tensor(x).to(device),
                get_batch(cur_offset)
            )
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     stock_num, alpha)
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[:, cur_offset - (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
        portfolio_perf = portfolio_backtest(cur_valid_pred, cur_valid_gt, cur_valid_mask, top_k=5)
    return loss, reg_loss, rank_loss, cur_valid_perf, portfolio_perf


def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = lookback_length
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1))


for epoch in range(epochs):
    print("epoch{}##########################################################".format(epoch + 1))
    np.random.shuffle(batch_offsets)
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0
    for j in range(valid_index - lookback_length - steps + 1):
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch(batch_offsets[j])
        )
        optimizer.zero_grad()
        prediction = model(data_batch)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                            stock_num, alpha)
        cur_loss = cur_loss
        cur_loss.backward()
        optimizer.step()

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()
    tra_loss = tra_loss / (valid_index - lookback_length - steps + 1)
    tra_reg_loss = tra_reg_loss / (valid_index - lookback_length - steps + 1)
    tra_rank_loss = tra_rank_loss / (valid_index - lookback_length - steps + 1)
    print('Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(tra_loss, tra_reg_loss, tra_rank_loss))

    val_loss, val_reg_loss, val_rank_loss, val_perf, val_portfolio = validate(valid_index, test_index)
    
    test_loss, test_reg_loss, test_rank_loss, test_perf, test_portfolio = validate(test_index, trade_dates)
    print(
        'Valid portfolio:',
        'mean_daily_return:{:.2e}, final_cum_return:{:.2e}, sharpe:{:.2e}'.format(
            val_portfolio['mean_daily_return'],
            val_portfolio['final_cumulative_return'],
            val_portfolio['sharpe']
        )
    )

    print(
        'Test portfolio:',
        'mean_daily_return:{:.2e}, final_cum_return:{:.2e}, sharpe:{:.2e}'.format(
            test_portfolio['mean_daily_return'],
            test_portfolio['final_cumulative_return'],
            test_portfolio['sharpe']
        )
    )

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        best_valid_perf = val_perf
        best_test_perf = test_perf

    print('Valid performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(val_perf['mse'], val_perf['IC'],
                                                     val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))
    print('Test performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(test_perf['mse'], test_perf['IC'],
                                                                            test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']), '\n\n')
