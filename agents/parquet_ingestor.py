"""
ParquetIngestor Agent Implementation
Handles financial data ingestion from Parquet files with schema validation
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Union

import pandas as pd


class ParquetIngestor:
    """
    Handles ingestion of financial data from Parquet files.
    Implements lightweight processing with strict schema validation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.schema_definitions = config.get('schema_definitions', {})
        self.validation_rules = config.get('validation_rules', {})
        self.chunk_size = config.get('chunk_size', 10000)
        self.max_memory_mb = config.get('max_memory_mb', 500)
        self.supported_schemas = ['market_data', 'order_flow', 'tick_data', 'ohlcv']
        self.ingestion_stats = {
            'files_processed': 0,
            'rows_processed': 0,
            'errors': 0,
            'last_ingestion': None
        }

    def validate_schema(self, df: pd.DataFrame, schema_type: str) -> Tuple[bool, List[str]]:
        """Validate dataframe against expected schema"""
        errors = []

        if schema_type not in self.supported_schemas:
            errors.append(f"Unsupported schema type: {schema_type}")
            return False, errors

        # Basic schema definitions
        schema_requirements = {
            'market_data': ['timestamp', 'symbol', 'price', 'volume'],
            'order_flow': ['timestamp', 'order_id', 'side', 'price', 'quantity'],
            'tick_data': ['timestamp', 'bid', 'ask', 'bid_size', 'ask_size'],
            'ohlcv': ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        }

        required_columns = schema_requirements.get(schema_type, [])
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Validate data types
        type_requirements = {
            'timestamp': ['datetime64', 'object'],
            'price': ['float64', 'int64'],
            'volume': ['float64', 'int64'],
            'quantity': ['float64', 'int64']
        }

        for col, expected_types in type_requirements.items():
            if col in df.columns:
                if str(df[col].dtype) not in expected_types:
                    errors.append(f"Column '{col}' has incorrect type: {df[col].dtype}")

        return len(errors) == 0, errors

    def preprocess_data(self, df: pd.DataFrame, schema_type: str) -> pd.DataFrame:
        """Preprocess data according to schema type"""
        # Convert timestamp columns
        timestamp_cols = ['timestamp']
        for col in timestamp_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception as e:
                    print(f"Warning: Could not convert {col} to datetime: {e}")

        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values based on schema
        if schema_type == 'market_data':
            df['volume'] = df['volume'].fillna(0)
        elif schema_type == 'ohlcv':
            # Forward fill price data
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill')

        return df

    def calculate_checksum(self, df: pd.DataFrame) -> str:
        """Calculate checksum for data integrity"""
        # Create a string representation of the dataframe
        df_string = df.to_json(orient='records', date_format='iso')
        return hashlib.sha256(df_string.encode()).hexdigest()

    def ingest_file(self, file_path: Union[str, Path], schema_type: str = 'market_data') -> Dict[str, Any]:
        """Ingest a single Parquet file"""
        file_path = Path(file_path)
        result = {
            'status': 'pending',
            'file': str(file_path),
            'schema_type': schema_type,
            'rows_processed': 0,
            'errors': [],
            'warnings': [],
            'checksum': None
        }

        try:
            # Check file exists
            if not file_path.exists():
                result['status'] = 'error'
                result['errors'].append(f"File not found: {file_path}")
                return result

            # Read parquet file
            df = pd.read_parquet(file_path)
            initial_rows = len(df)

            # Validate schema
            is_valid, validation_errors = self.validate_schema(df, schema_type)
            if not is_valid:
                result['status'] = 'error'
                result['errors'].extend(validation_errors)
                return result

            # Preprocess data
            df = self.preprocess_data(df, schema_type)

            # Check memory usage
            memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            if memory_usage_mb > self.max_memory_mb:
                result['warnings'].append(
                    f"File exceeds memory limit ({memory_usage_mb:.2f}MB > {self.max_memory_mb}MB)")
                # Process in chunks
                result['processing_mode'] = 'chunked'

            # Calculate checksum
            result['checksum'] = self.calculate_checksum(df)

            # Update stats
            self.ingestion_stats['files_processed'] += 1
            self.ingestion_stats['rows_processed'] += len(df)
            self.ingestion_stats['last_ingestion'] = datetime.now().isoformat()

            result['status'] = 'success'
            result['rows_processed'] = len(df)
            result['data_summary'] = {
                'initial_rows': initial_rows,
                'final_rows': len(df),
                'columns': list(df.columns),
                'memory_usage_mb': memory_usage_mb,
                'date_range': {
                    'start': str(df['timestamp'].min()) if 'timestamp' in df.columns else None,
                    'end': str(df['timestamp'].max()) if 'timestamp' in df.columns else None
                }
            }

            # Store processed data for pipeline
            result['dataframe'] = df

        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(f"Ingestion failed: {str(e)}")
            self.ingestion_stats['errors'] += 1

        return result

    def ingest_batch(self, file_paths: List[Union[str, Path]], schema_type: str = 'market_data') -> Dict[str, Any]:
        """Ingest multiple Parquet files"""
        batch_result = {
            'status': 'pending',
            'total_files': len(file_paths),
            'successful': 0,
            'failed': 0,
            'results': []
        }

        for file_path in file_paths:
            file_result = self.ingest_file(file_path, schema_type)
            batch_result['results'].append(file_result)

            if file_result['status'] == 'success':
                batch_result['successful'] += 1
            else:
                batch_result['failed'] += 1

        batch_result['status'] = 'complete'
        return batch_result

    def stream_ingest(self, file_path: Union[str, Path], schema_type: str = 'market_data',
                      callback=None) -> Dict[str, Any]:
        """Stream ingest large files in chunks"""
        file_path = Path(file_path)
        result = {
            'status': 'streaming',
            'file': str(file_path),
            'chunks_processed': 0,
            'total_rows': 0
        }

        try:
            # Use Parquet file iterator for chunked reading
            parquet_file = pd.read_parquet(file_path, engine='pyarrow')
            total_rows = len(parquet_file)

            # Process in chunks
            for i in range(0, total_rows, self.chunk_size):
                chunk = parquet_file.iloc[i:i + self.chunk_size]

                # Validate and preprocess chunk
                is_valid, errors = self.validate_schema(chunk, schema_type)
                if is_valid:
                    chunk = self.preprocess_data(chunk, schema_type)

                    # Call callback if provided
                    if callback:
                        callback(chunk, i, total_rows)

                    result['chunks_processed'] += 1
                    result['total_rows'] += len(chunk)

            result['status'] = 'success'

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get current ingestion statistics"""
        return self.ingestion_stats.copy()

    def reset_stats(self):
        """Reset ingestion statistics"""
        self.ingestion_stats = {
            'files_processed': 0,
            'rows_processed': 0,
            'errors': 0,
            'last_ingestion': None
        }
