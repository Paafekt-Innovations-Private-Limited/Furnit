package com.furnit.android.auth

import java.util.Locale

/**
 * Country code data for phone number input
 * Matches iOS LoginView country list
 */
data class CountryCode(
    val name: String,
    val dialCode: String,
    val code: String,
    val flag: String
) {
    val displayName: String get() = "$flag $name ($dialCode)"
    val shortDisplay: String get() = "$flag $dialCode"

    companion object {
        val countries = listOf(
            CountryCode("United States", "+1", "US", "\uD83C\uDDFA\uD83C\uDDF8"),
            CountryCode("United Kingdom", "+44", "GB", "\uD83C\uDDEC\uD83C\uDDE7"),
            CountryCode("Canada", "+1", "CA", "\uD83C\uDDE8\uD83C\uDDE6"),
            CountryCode("Australia", "+61", "AU", "\uD83C\uDDE6\uD83C\uDDFA"),
            CountryCode("India", "+91", "IN", "\uD83C\uDDEE\uD83C\uDDF3"),
            CountryCode("Germany", "+49", "DE", "\uD83C\uDDE9\uD83C\uDDEA"),
            CountryCode("France", "+33", "FR", "\uD83C\uDDEB\uD83C\uDDF7"),
            CountryCode("Italy", "+39", "IT", "\uD83C\uDDEE\uD83C\uDDF9"),
            CountryCode("Spain", "+34", "ES", "\uD83C\uDDEA\uD83C\uDDF8"),
            CountryCode("Brazil", "+55", "BR", "\uD83C\uDDE7\uD83C\uDDF7"),
            CountryCode("Mexico", "+52", "MX", "\uD83C\uDDF2\uD83C\uDDFD"),
            CountryCode("Japan", "+81", "JP", "\uD83C\uDDEF\uD83C\uDDF5"),
            CountryCode("South Korea", "+82", "KR", "\uD83C\uDDF0\uD83C\uDDF7"),
            CountryCode("China", "+86", "CN", "\uD83C\uDDE8\uD83C\uDDF3"),
            CountryCode("Russia", "+7", "RU", "\uD83C\uDDF7\uD83C\uDDFA"),
            CountryCode("South Africa", "+27", "ZA", "\uD83C\uDDFF\uD83C\uDDE6"),
            CountryCode("Nigeria", "+234", "NG", "\uD83C\uDDF3\uD83C\uDDEC"),
            CountryCode("Egypt", "+20", "EG", "\uD83C\uDDEA\uD83C\uDDEC"),
            CountryCode("Kenya", "+254", "KE", "\uD83C\uDDF0\uD83C\uDDEA"),
            CountryCode("Saudi Arabia", "+966", "SA", "\uD83C\uDDF8\uD83C\uDDE6"),
            CountryCode("UAE", "+971", "AE", "\uD83C\uDDE6\uD83C\uDDEA"),
            CountryCode("Israel", "+972", "IL", "\uD83C\uDDEE\uD83C\uDDF1"),
            CountryCode("Turkey", "+90", "TR", "\uD83C\uDDF9\uD83C\uDDF7"),
            CountryCode("Poland", "+48", "PL", "\uD83C\uDDF5\uD83C\uDDF1"),
            CountryCode("Netherlands", "+31", "NL", "\uD83C\uDDF3\uD83C\uDDF1"),
            CountryCode("Belgium", "+32", "BE", "\uD83C\uDDE7\uD83C\uDDEA"),
            CountryCode("Sweden", "+46", "SE", "\uD83C\uDDF8\uD83C\uDDEA"),
            CountryCode("Norway", "+47", "NO", "\uD83C\uDDF3\uD83C\uDDF4"),
            CountryCode("Denmark", "+45", "DK", "\uD83C\uDDE9\uD83C\uDDF0"),
            CountryCode("Finland", "+358", "FI", "\uD83C\uDDEB\uD83C\uDDEE"),
            CountryCode("Austria", "+43", "AT", "\uD83C\uDDE6\uD83C\uDDF9"),
            CountryCode("Switzerland", "+41", "CH", "\uD83C\uDDE8\uD83C\uDDED"),
            CountryCode("Portugal", "+351", "PT", "\uD83C\uDDF5\uD83C\uDDF9"),
            CountryCode("Greece", "+30", "GR", "\uD83C\uDDEC\uD83C\uDDF7"),
            CountryCode("Ireland", "+353", "IE", "\uD83C\uDDEE\uD83C\uDDEA"),
            CountryCode("New Zealand", "+64", "NZ", "\uD83C\uDDF3\uD83C\uDDFF"),
            CountryCode("Singapore", "+65", "SG", "\uD83C\uDDF8\uD83C\uDDEC"),
            CountryCode("Malaysia", "+60", "MY", "\uD83C\uDDF2\uD83C\uDDFE"),
            CountryCode("Thailand", "+66", "TH", "\uD83C\uDDF9\uD83C\uDDED"),
            CountryCode("Indonesia", "+62", "ID", "\uD83C\uDDEE\uD83C\uDDE9"),
            CountryCode("Philippines", "+63", "PH", "\uD83C\uDDF5\uD83C\uDDED"),
            CountryCode("Vietnam", "+84", "VN", "\uD83C\uDDFB\uD83C\uDDF3"),
            CountryCode("Pakistan", "+92", "PK", "\uD83C\uDDF5\uD83C\uDDF0"),
            CountryCode("Bangladesh", "+880", "BD", "\uD83C\uDDE7\uD83C\uDDE9"),
            CountryCode("Argentina", "+54", "AR", "\uD83C\uDDE6\uD83C\uDDF7"),
            CountryCode("Chile", "+56", "CL", "\uD83C\uDDE8\uD83C\uDDF1"),
            CountryCode("Colombia", "+57", "CO", "\uD83C\uDDE8\uD83C\uDDF4"),
            CountryCode("Peru", "+51", "PE", "\uD83C\uDDF5\uD83C\uDDEA"),
            CountryCode("Venezuela", "+58", "VE", "\uD83C\uDDFB\uD83C\uDDEA"),
            CountryCode("Ukraine", "+380", "UA", "\uD83C\uDDFA\uD83C\uDDE6"),
            CountryCode("Czech Republic", "+420", "CZ", "\uD83C\uDDE8\uD83C\uDDFF"),
            CountryCode("Romania", "+40", "RO", "\uD83C\uDDF7\uD83C\uDDF4"),
            CountryCode("Hungary", "+36", "HU", "\uD83C\uDDED\uD83C\uDDFA"),
            CountryCode("Morocco", "+212", "MA", "\uD83C\uDDF2\uD83C\uDDE6"),
            CountryCode("Algeria", "+213", "DZ", "\uD83C\uDDE9\uD83C\uDDFF"),
            CountryCode("Tunisia", "+216", "TN", "\uD83C\uDDF9\uD83C\uDDF3"),
            CountryCode("Ghana", "+233", "GH", "\uD83C\uDDEC\uD83C\uDDED"),
            CountryCode("Tanzania", "+255", "TZ", "\uD83C\uDDF9\uD83C\uDDFF"),
            CountryCode("Ethiopia", "+251", "ET", "\uD83C\uDDEA\uD83C\uDDF9"),
            CountryCode("Hong Kong", "+852", "HK", "\uD83C\uDDED\uD83C\uDDF0"),
            CountryCode("Taiwan", "+886", "TW", "\uD83C\uDDF9\uD83C\uDDFC"),
            CountryCode("Sri Lanka", "+94", "LK", "\uD83C\uDDF1\uD83C\uDDF0"),
            CountryCode("Nepal", "+977", "NP", "\uD83C\uDDF3\uD83C\uDDF5"),
            CountryCode("Qatar", "+974", "QA", "\uD83C\uDDF6\uD83C\uDDE6"),
            CountryCode("Kuwait", "+965", "KW", "\uD83C\uDDF0\uD83C\uDDFC")
        )

        /**
         * Get country code based on device locale
         */
        fun getDefaultCountry(): CountryCode {
            val locale = Locale.getDefault()
            val countryCode = locale.country.uppercase()
            return countries.find { it.code == countryCode } ?: countries[0] // Default to US
        }

        /**
         * Search countries by name or dial code
         */
        fun search(query: String): List<CountryCode> {
            if (query.isBlank()) return countries
            val lowerQuery = query.lowercase()
            return countries.filter {
                it.name.lowercase().contains(lowerQuery) ||
                it.dialCode.contains(query) ||
                it.code.lowercase().contains(lowerQuery)
            }
        }
    }
}
