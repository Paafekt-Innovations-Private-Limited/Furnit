*"----------------------------------------------------------------------
*"* Wrapper for legacy FM CONVERSION_EXIT_ALPHA_INPUT
*"* Purpose: alpha conversion input
*"* Modern API: CONVERSION_EXIT_ALPHA_INPUT (FM (wrap))
*"----------------------------------------------------------------------
CLASS zcl_zsample_conversion_exit_alpha_input_wrapper DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS call
      IMPORTING
        iv_fm_name TYPE string DEFAULT 'CONVERSION_EXIT_ALPHA_INPUT'
      RETURNING
        VALUE(rv_success) TYPE abap_bool.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS zcl_zsample_conversion_exit_alpha_input_wrapper IMPLEMENTATION.
  METHOD call.
    CALL FUNCTION iv_fm_name
      EXCEPTIONS
        OTHERS = 1.
    rv_success = xsdbool( sy-subrc = 0 ).
  ENDMETHOD.
ENDCLASS.
