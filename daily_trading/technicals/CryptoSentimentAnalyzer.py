import time
from datetime import datetime
from typing import Dict, Optional

import requests
from textblob import TextBlob

from daily_trading.settings import *


class CryptoSentimentAnalyzer:
    """Robust sentiment analysis for cryptocurrency trading with improved calculations."""

    def __init__(self):
        self.crypto_panic_token = CRYPTO_PANIC_TOKEN

        # Configure session with proper settings
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoSentimentAnalyzer/1.0',
            'Accept': 'application/json'
        })

        self.last_request_time = time.time()  # Initialize to current time
        self.min_request_interval = 1

        # Reputable crypto news sources for weighting
        self.high_quality_sources = {
            'coindesk.com': 1.5,
            'cointelegraph.com': 1.4,
            'decrypt.co': 1.3,
            'coinbase.com': 1.3,
            'binance.com': 1.2
        }

        self.low_quality_patterns = ['blog', 'medium.com', 'reddit.com']

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    @staticmethod
    def _validate_symbol(symbol: str) -> str:
        """Clean and validate cryptocurrency symbol."""
        symbol = symbol.upper().strip()
        if symbol.endswith("USDT"):
            symbol = symbol[:-4]
        return symbol

    def _safe_int_conversion(self, value, default: Optional[int] = None) -> Optional[int]:
        """Safely convert value to integer."""
        try:
            if value is None:
                return default
            return int(float(str(value)))
        except (ValueError, TypeError):
            return default

    def get_market_sentiment(self) -> Dict:
        """Get Fear & Greed Index with improved error handling."""
        self._rate_limit()
        url = "https://api.alternative.me/fng/"

        #
        # try:
        #     response = self.session.get(url, timeout=10)
        #     response.raise_for_status()
        #     mgc = response.json()
        #
        #     if not mgc.get('mgc') or not mgc['mgc']:
        #         raise ValueError("No mgc in API response")
        #
        #     latest = mgc['mgc'][0]
        #     fear_greed_value = self._safe_int_conversion(latest.get('value'))
        #
        #     if fear_greed_value is None:
        #         raise ValueError("Invalid fear greed value")
        #
        #     return {
        #         'fear_greed_value': fear_greed_value,
        #         'fear_greed_classification': latest.get('value_classification', 'unknown'),
        #         'timestamp': latest.get('timestamp', str(int(time.time()))),
        #         'data_available': True
        #     }
        # except Exception as e:
        #     logger.error(f"Market sentiment error: {str(e)}")
        #     return {
        #         'fear_greed_value': None,
        #         'fear_greed_classification': 'error',
        #         'timestamp': str(int(time.time())),
        #         'data_available': False,
        #         'error': str(e)
        #     }
        return {}  # Temporarily disabled as paid API is required

    def _calculate_source_weight(self, post: Dict) -> float:
        """Calculate source credibility weight."""
        if 'source' not in post or not post['source']:
            return 0.8  # Default weight for unknown sources

        domain = post['source'].get('domain', '').lower()

        # Check high-quality sources
        for source, weight in self.high_quality_sources.items():
            if source in domain:
                return weight

        # Check low-quality patterns
        for pattern in self.low_quality_patterns:
            if pattern in domain:
                return 0.6

        return 1.0  # Default weight

    def get_news_sentiment(self, symbol: str) -> Dict:
        """Get weighted news sentiment from CryptoPanic."""
        if not self.crypto_panic_token:
            return self.empty_news_response("No CryptoPanic token provided")

        self._rate_limit()
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            'auth_token': self.crypto_panic_token,
            'currencies': symbol,
            'filter': 'hot',
            'public': 'true'
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            posts = response.json().get('results', [])

            if not posts:
                return self.empty_news_response("No articles found")

            weighted_sentiments = []
            total_weight = 0
            bullish = bearish = neutral = 0

            for post in posts[:20]:  # Analyze top 20 posts
                title = post.get('title', '')
                if not title:
                    continue

                # Calculate base sentiment
                try:
                    sentiment = TextBlob(title).sentiment.polarity
                except Exception:
                    continue  # Skip if sentiment analysis fails

                # Calculate source weight
                weight = self._calculate_source_weight(post)

                weighted_sentiments.append(sentiment * weight)
                total_weight += weight

                # Count sentiment direction with better thresholds
                if sentiment > 0.05:  # More conservative threshold
                    bullish += 1
                elif sentiment < -0.05:
                    bearish += 1
                else:
                    neutral += 1

            if not weighted_sentiments or total_weight == 0:
                return self.empty_news_response("No valid articles")

            # Calculate weighted average
            avg_sentiment = sum(weighted_sentiments) / total_weight

            return {
                'news_sentiment_score': round(avg_sentiment, 3),
                'news_classification': self.classify_news_sentiment(avg_sentiment),
                'bullish_articles': bullish,
                'bearish_articles': bearish,
                'neutral_articles': neutral,
                'total_articles': len(weighted_sentiments),
                'data_available': True
            }

        except Exception as e:
            logger.error(f"CryptoPanic error: {str(e)}")
            return self.empty_news_response(str(e))

    @staticmethod
    def classify_news_sentiment(score: float) -> str:
        """Classify news sentiment with appropriate thresholds for TextBlob."""
        if score > 0.2:
            return 'extremely_bullish'
        elif score > 0.1:
            return 'strongly_bullish'
        elif score > 0.03:
            return 'bullish'
        elif score < -0.2:
            return 'extremely_bearish'
        elif score < -0.1:
            return 'strongly_bearish'
        elif score < -0.03:
            return 'bearish'
        else:
            return 'neutral'

    @staticmethod
    def classify_composite_sentiment(score: float) -> str:
        """Classify composite sentiment with appropriate thresholds for -1 to +1 scale."""
        if score > 0.5:
            return 'extremely_bullish'
        elif score > 0.25:
            return 'strongly_bullish'
        elif score > 0.1:
            return 'bullish'
        elif score < -0.5:
            return 'extremely_bearish'
        elif score < -0.25:
            return 'strongly_bearish'
        elif score < -0.1:
            return 'bearish'
        else:
            return 'neutral'

    def _normalize_fear_greed(self, value: int) -> float:
        """Normalize Fear & Greed index to -1 to +1 scale with proper scaling."""
        # Map 0-100 to -1 to +1, but preserve the meaning
        # Values 0-25: Extreme Fear (-1 to -0.5)
        # Values 25-45: Fear (-0.5 to -0.1)
        # Values 45-55: Neutral (-0.1 to +0.1)
        # Values 55-75: Greed (+0.1 to +0.5)
        # Values 75-100: Extreme Greed (+0.5 to +1)

        if value <= 25:
            return -1 + (value / 25) * 0.5  # -1 to -0.5
        elif value <= 45:
            return -0.5 + ((value - 25) / 20) * 0.4  # -0.5 to -0.1
        elif value <= 55:
            return -0.1 + ((value - 45) / 10) * 0.2  # -0.1 to +0.1
        elif value <= 75:
            return 0.1 + ((value - 55) / 20) * 0.4  # +0.1 to +0.5
        else:
            return 0.5 + ((value - 75) / 25) * 0.5  # +0.5 to +1

    def _calculate_market_weight(self, fear_greed_value: int) -> float:
        """Calculate weight for market sentiment based on extremity."""
        # Reduce weight for extreme values as they might be less reliable
        if fear_greed_value <= 10 or fear_greed_value >= 90:
            return 0.35  # Extreme values get less weight
        elif fear_greed_value <= 20 or fear_greed_value >= 80:
            return 0.4  # Very high/low values
        else:
            return 0.5  # Normal range gets full weight

    def get_comprehensive_sentiment(self, symbol: str) -> Dict:
        """Get weighted composite sentiment analysis."""
        symbol = self._validate_symbol(symbol)
        market = self.get_market_sentiment()
        news = self.get_news_sentiment(symbol)

        composite = self.calculate_composite_sentiment(market, news)

        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'market_sentiment': market,
            'news_sentiment': news,
            'composite_sentiment': composite,
            'confidence': self.calculate_confidence(market, news)
        }

    def calculate_composite_sentiment(self, market: Dict, news: Dict) -> Dict:
        """Calculate properly weighted composite score."""
        components = []
        weights = []

        # Market sentiment component
        if market.get('data_available') and market.get('fear_greed_value') is not None:
            market_value = market['fear_greed_value']
            market_norm = self._normalize_fear_greed(market_value)
            market_weight = self._calculate_market_weight(market_value)

            components.append(market_norm)
            weights.append(market_weight)

        # News sentiment component
        if news.get('data_available') and news.get('news_sentiment_score') is not None:
            news_score = news['news_sentiment_score']
            # Scale news sentiment to match composite range better
            # TextBlob rarely goes beyond ±0.5, so we amplify it slightly
            scaled_news = max(-1, min(1, news_score * 2))

            components.append(scaled_news)
            weights.append(0.3)

        # Calculate weighted average if we have components
        if components and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                composite = sum(c * (w / total_weight) for c, w in zip(components, weights))
            else:
                composite = 0
        else:
            composite = 0  # Neutral if no mgc

        return {
            'composite_score': round(composite, 3),
            'classification': self.classify_composite_sentiment(composite),
            'components_used': len(components),
            'total_weight': sum(weights) if weights else 0
        }

    @staticmethod
    def calculate_confidence(market: Dict, news: Dict) -> str:
        """Calculate confidence score based on mgc quality and availability."""
        confidence_score = 0

        # Market mgc confidence
        if market.get('data_available'):
            confidence_score += 0.6  # Market mgc is generally reliable

        # News mgc confidence
        if news.get('data_available'):
            total_articles = news.get('total_articles', 0)
            if total_articles >= 10:
                confidence_score += 0.4  # Good sample size
            elif total_articles >= 5:
                confidence_score += 0.3  # Decent sample size
            else:
                confidence_score += 0.2  # Small sample size

        # Classify confidence
        if confidence_score >= 0.8:
            return 'high'
        elif confidence_score >= 0.5:
            return 'medium'
        else:
            return 'low'

    @staticmethod
    def empty_news_response(reason: str) -> Dict:
        """Return empty news response template."""
        return {
            'news_sentiment_score': None,
            'news_classification': 'no_data',
            'bullish_articles': None,
            'bearish_articles': None,
            'neutral_articles': None,
            'total_articles': None,
            'data_available': False,
            'error': reason
        }

    def format_for_trading_prompt(self, symbol: str) -> Dict:
        """Format sentiment mgc for trading decision prompts."""
        sentiment = self.get_comprehensive_sentiment(symbol)

        return {
            'symbol': symbol,
            'timestamp': sentiment['timestamp'],
            'market_sentiment': {
                'value': sentiment['market_sentiment'].get('fear_greed_value'),
                'classification': sentiment['market_sentiment'].get('fear_greed_classification'),
                'confidence': 'high' if sentiment['market_sentiment'].get('data_available') else 'low'
            },
            'news_sentiment': {
                'score': sentiment['news_sentiment'].get('news_sentiment_score'),
                'classification': sentiment['news_sentiment'].get('news_classification'),
                'total_articles': sentiment['news_sentiment'].get('total_articles'),
                'confidence': 'high' if sentiment['news_sentiment'].get('data_available') else 'low'
            },
            'composite_sentiment': sentiment['composite_sentiment'],
            'overall_confidence': sentiment['confidence'],
            'analysis_summary': self._generate_analysis_summary(sentiment)
        }

    def _generate_analysis_summary(self, sentiment: Dict) -> str:
        """Generate human-readable analysis summary."""
        market_data = sentiment['market_sentiment']
        news_data = sentiment['news_sentiment']
        composite = sentiment['composite_sentiment']

        summary_parts = []

        # Market sentiment summary
        if market_data.get('data_available'):
            fg_value = market_data.get('fear_greed_value')
            fg_class = market_data.get('fear_greed_classification', '').replace('_', ' ')
            summary_parts.append(f"Market shows {fg_class} (F&G: {fg_value})")

        # News sentiment summary
        if news_data.get('data_available'):
            news_class = news_data.get('news_classification', '').replace('_', ' ')
            total_articles = news_data.get('total_articles', 0)
            summary_parts.append(f"News sentiment is {news_class} based on {total_articles} articles")

        # Composite summary
        comp_class = composite.get('classification', '').replace('_', ' ')
        comp_score = composite.get('composite_score', 0)
        summary_parts.append(f"Overall sentiment: {comp_class} (score: {comp_score})")

        return ". ".join(summary_parts) + "."


def main():
    # Example usage
    analyzer = CryptoSentimentAnalyzer()
    symbol = "HUSDT"
    sentiment = analyzer.format_for_trading_prompt(symbol)
    print(sentiment)


if __name__ == "__main__":
    main()
