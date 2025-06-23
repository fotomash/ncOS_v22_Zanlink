# Pipeline Data Models

This document describes the typed objects exchanged between the unified engines.

## DataRequest
`DataRequest` represents a symbol and list of timeframes to fetch.  
Fields:
- `symbol` (`str`)
- `timeframes` (`List[str]`)

## DataResult
Returned by the data pipeline.
- `status` (`str`)
- `symbol` (`str`)
- `data` (`Dict[str, Any]`)
- `timestamp` (`datetime`)

## AnalysisResult
Produced by the analysis engine.
- `status` (`str`)
- `symbol` (`str`)
- `analysis` (`Dict[str, Any]`)
- `confluence_score` (`float`)
- `timestamp` (`datetime`)

## StrategyMatch
Result of strategy matching.
- `strategy` (`Optional[str]`)
- `confidence` (`float`)
- `status` (`str`)

## ExecutionResult
Returned by the execution engine when an order is generated.
- `status` (`str`)
- `strategy` (`Optional[str]`)
- `symbol` (`Optional[str]`)
- `entry_price` (`Optional[float]`)
- `stop_loss` (`Optional[float]`)
- `take_profit` (`Optional[float]`)
- `position_size` (`Optional[float]`)
- `direction` (`Optional[str]`)
- `timestamp` (`datetime`)
