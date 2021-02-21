defmodule TryNxTest do
  use ExUnit.Case
  doctest TryNx

  test "train model" do
    assert TryNx.now() == :ok
  end
end
