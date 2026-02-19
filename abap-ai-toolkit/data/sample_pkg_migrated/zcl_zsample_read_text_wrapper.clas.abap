*"----------------------------------------------------------------------
*"* Wrapper for legacy FM READ_TEXT
*"* Purpose: read text
*"* Modern API: I_*Text RAP (RAP BO)
*"----------------------------------------------------------------------
CLASS zcl_zsample_read_text_wrapper DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS call
      IMPORTING
        iv_fm_name TYPE string DEFAULT 'READ_TEXT'
      RETURNING
        VALUE(rv_success) TYPE abap_bool.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS zcl_zsample_read_text_wrapper IMPLEMENTATION.
  METHOD call.
    CALL FUNCTION iv_fm_name
      EXCEPTIONS
        OTHERS = 1.
    rv_success = xsdbool( sy-subrc = 0 ).
  ENDMETHOD.
ENDCLASS.
