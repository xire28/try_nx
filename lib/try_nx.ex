defmodule TryNx do
  @moduledoc """
  Documentation for `TryNx`.
  """

  @doc """
  Train model using MSE and SGD
  Reference: https://jovian.ai/aakashns/02-linear-regression (pytorch version)

  ## Examples

      iex> TryNx.now()
      :ok

  """

  require Logger

  import Nx.Defn

  @max_epoch 10000
  @max_loss 0.5

  def now do
    inputs =
      Nx.tensor(
        [
          [73, 67, 43],
          [91, 88, 64],
          [87, 134, 58],
          [102, 43, 37],
          [69, 96, 70]
        ],
        type: {:f, 32}
      )

    targets =
      Nx.tensor(
        [
          [56, 70],
          [81, 101],
          [119, 133],
          [22, 37],
          [103, 119]
        ],
        type: {:f, 32}
      )

    weights = Nx.random_uniform({2, 3})
    biases = Nx.random_uniform({2})

    {preds, weights, biases, epoch, loss} = train(inputs, targets, weights, biases)

    Logger.info("""
      Training result:
      epoch: #{inspect(epoch)}
      loss: #{inspect(loss)}
      preds: #{inspect(preds)}
      targets: #{inspect(targets)}}
      weights: #{inspect(biases)}
      biases: #{inspect(weights)}
    """)

    :ok
  end

  def train(inputs, targets, weights, biases, epoch \\ 0, loss \\ :infinity)

  def train(inputs, targets, weights, biases, epoch, loss) when epoch < @max_epoch or loss < @max_loss do
    Logger.info("""
      Training step:
      epoch: #{inspect(epoch)}
      loss: #{inspect(loss)}
      weights: #{inspect(weights)}
      biases: #{inspect(biases)}
    """)

    {loss, {weight_grads, bias_grads}} = eval(inputs, targets, weights, biases)
    loss = Nx.to_scalar(loss)

    weights = sgd(weights, weight_grads)
    biases = sgd(biases, bias_grads)
    train(inputs, targets, weights, biases, epoch + 1, loss)
  end

  def train(inputs, _targets, weights, biases, epoch, loss),
    do: {model(inputs, weights, biases), weights, biases, epoch, loss}

  defn eval(inputs, targets, weights, biases) do
    loss =
      model(inputs, weights, biases)
      |> mse(targets)

    {loss, grad({weights, biases}, loss)}
  end

  defn model(inputs, weights, biases) do
    inputs
    |> Nx.dot(Nx.transpose(weights))
    |> Nx.add(biases)
  end

  defn mse(left, right) do
    left
    |> Nx.subtract(right)
    |> Nx.power(2)
    |> Nx.mean()
  end

  defn sgd(tensor, grads, step \\ 1.0e-5) do
    tensor
    |> Nx.subtract(Nx.multiply(grads, step))
  end
end
